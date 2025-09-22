#!/usr/bin/env python3
"""
Video background remover / silhouette generator (macOS Vision if available, else rembg)
with progress + verbose logging.

- Default: remove background, keep original subject colors.
  * When writing PNG frames: transparent background (alpha).
  * When writing a video: composited over a solid background (default black).
- --sil: make silhouettes instead of keeping colors
- --sil-color / -c: pick silhouette color (default: black)
- --bg / -b: background fill color (for frames = transparent by default; for video = black by default)
- --thresh / -t: optional hard threshold (otherwise uses soft mask)
- --frames: write processed PNG frames to a directory instead of a video
- --keep-audio: try to copy original audio stream (needs ffmpeg)
- --verbose / -v: print extra details about steps/backends
- Shows a tqdm progress bar if available

Examples:
  python vid_sil.py in.mp4 -o out.mp4
  python vid_sil.py in.mp4 -o out.mp4 -s -c "#00aaff" -b white
  python vid_sil.py in.mp4 -o out_frames --frames
  python vid_sil.py in.mp4 -o out.mp4 -b white -t 0.6 --keep-audio -v
"""

import argparse, pathlib, sys, ctypes, shutil, subprocess, time
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import cv2

# optional tqdm
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ----------------------- macOS Vision (optional) -----------------------
def _try_import_vision():
    try:
        import Vision as VN
        import Quartz as Q
        from Foundation import NSURL
        from Quartz import CoreVideo as CV
        return VN, Q, NSURL, CV
    except Exception:
        return None

def _load_cgimage(Q, NSURL, path: str):
    url = NSURL.fileURLWithPath_(path)
    src = Q.CGImageSourceCreateWithURL(url, None)
    if not src:
        raise RuntimeError(f"Failed to open image: {path}")
    cg = Q.CGImageSourceCreateImageAtIndex(src, 0, None)
    if not cg:
        raise RuntimeError(f"Failed to decode first frame: {path}")
    return cg

def _cgsize(Q, cg):
    return Q.CGImageGetWidth(cg), Q.CGImageGetHeight(cg)

def _cgimage_to_gray(Q, cg) -> np.ndarray:
    """Render CGImage into 8-bit grayscale numpy array (H, W)."""
    w, h = _cgsize(Q, cg)
    cs = Q.CGColorSpaceCreateDeviceGray()
    bpr = w  # 1 byte per pixel
    ctx = Q.CGBitmapContextCreate(None, w, h, 8, bpr, cs, 0)  # no alpha
    if not ctx:
        raise RuntimeError("Failed to create grayscale bitmap context")
    Q.CGContextDrawImage(ctx, Q.CGRectMake(0, 0, w, h), cg)
    ptr = Q.CGBitmapContextGetData(ctx)
    if not ptr:
        raise RuntimeError("Failed to get bitmap data")
    buf = ctypes.string_at(int(ptr), bpr * h)
    return np.frombuffer(buf, dtype=np.uint8).reshape(h, bpr)[:, :w]

def _pixelbuffer_to_u8(CV, pb) -> np.ndarray:
    CV.CVPixelBufferLockBaseAddress(pb, 0)
    try:
        base = CV.CVPixelBufferGetBaseAddress(pb)
        bpr  = CV.CVPixelBufferGetBytesPerRow(pb)
        w    = CV.CVPixelBufferGetWidth(pb)
        h    = CV.CVPixelBufferGetHeight(pb)
        buf  = ctypes.string_at(int(base), bpr * h)
        return np.frombuffer(buf, dtype=np.uint8).reshape(h, bpr)[:, :w]
    finally:
        CV.CVPixelBufferUnlockBaseAddress(pb, 0)

def _vision_mask_for_rgb(rgb_frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Take an RGB frame (H,W,3 uint8), save to a temp PNG for CGImage,
    run Vision to produce a soft mask [0..1]. Returns None if Vision isn't available.
    """
    imported = _try_import_vision()
    if not imported:
        return None
    VN, Q, NSURL, CV = imported

    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(suffix=".png") as tf:
        # Avoid Pillow 'mode' kwarg deprecation by not passing it
        Image.fromarray(rgb_frame).save(tf.name)

        try:
            cg = _load_cgimage(Q, NSURL, tf.name)
            w, h = _cgsize(Q, cg)

            handler = VN.VNImageRequestHandler.imageRequestHandlerWithCGImage_options_(cg, {})

            # Try foreground instance mask (macOS 14+)
            try:
                if hasattr(VN, "VNGenerateForegroundInstanceMaskRequest"):
                    req = None
                    for ctor in (
                        lambda: VN.VNGenerateForegroundInstanceMaskRequest.alloc().initWithRevision_error_(1, None),
                        lambda: VN.VNGenerateForegroundInstanceMaskRequest.request(),
                        lambda: VN.VNGenerateForegroundInstanceMaskRequest.alloc().init(),
                    ):
                        try:
                            req = ctor()
                            if req is not None:
                                break
                        except Exception:
                            req = None
                    if req is not None:
                        ok = handler.performRequests_error_([req], None)
                        if ok and req.results():
                            obs = req.results()[0]
                            size = Q.CGSizeMake(w, h)
                            cg_mask, _ = obs.generateScaledMaskForImageOfSize_error_(size, None)
                            if cg_mask:
                                return _cgimage_to_gray(Q, cg_mask).astype(np.float32) / 255.0
            except Exception:
                pass

            # Fallback: person segmentation (macOS 12–13)
            try:
                req = None
                for ctor in (
                    lambda: VN.VNGeneratePersonSegmentationRequest.alloc().init(),
                    lambda: VN.VNGeneratePersonSegmentationRequest.request(),
                ):
                    try:
                        req = ctor()
                        if req is not None:
                            break
                    except Exception:
                        req = None
                if req is None:
                    return None

                if hasattr(VN, "VNGeneratePersonSegmentationRequestQualityLevelAccurate"):
                    req.setQualityLevel_(VN.VNGeneratePersonSegmentationRequestQualityLevelAccurate)
                req.setOutputPixelFormat_(0x00000041)  # kCVPixelFormatType_OneComponent8

                ok = handler.performRequests_error_([req], None)
                if not ok or not req.results():
                    return None
                pb = req.results()[0].pixelBuffer()
                return _pixelbuffer_to_u8(CV, pb).astype(np.float32) / 255.0
            except Exception:
                return None
        except Exception:
            return None


# ----------------------- rembg fallback -----------------------
_REMBG_SESSION = None

def _get_rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        from rembg import new_session
        _REMBG_SESSION = new_session()
    return _REMBG_SESSION

def _rembg_mask_for_rgb(rgb_frame: np.ndarray) -> np.ndarray:
    """
    Returns a SOFT alpha mask [0..1] using rembg for an RGB ndarray frame.
    """
    from rembg import remove
    import io
    pil = Image.fromarray(rgb_frame).convert("RGBA")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    rgba = Image.open(io.BytesIO(remove(buf.getvalue(), session=_get_rembg_session()))).convert("RGBA")
    alpha = np.array(rgba)[:, :, 3].astype(np.float32) / 255.0
    return alpha


# ----------------------- helpers -----------------------
def _parse_color(s: Optional[str]) -> Optional[Tuple[float, float, float]]:
    if s is None:
        return None
    s = s.strip().lower()
    if s in ("white", "#ffffff"):
        return (1.0, 1.0, 1.0)
    if s in ("black", "#000000"):
        return (0.0, 0.0, 0.0)
    if s.startswith("#") and len(s) == 7:
        r = int(s[1:3], 16) / 255.0
        g = int(s[3:5], 16) / 255.0
        b = int(s[5:7], 16) / 255.0
        return (r, g, b)
    if "," in s:
        r, g, b = [float(x.strip()) for x in s.split(",")]
        if max(r, g, b) > 1.0:  # 0–255 inputs
            r, g, b = r / 255.0, g / 255.0, b / 255.0
        return (r, g, b)
    raise ValueError(f"Unrecognized color: {s}")

def _get_soft_mask(rgb_frame: np.ndarray, backend_hint: Optional[str] = None):
    """
    backend_hint: 'vision' or 'rembg' or None (auto).
    Returns (mask, backend_used)
    """
    if backend_hint == "vision":
        m = _vision_mask_for_rgb(rgb_frame)
        if m is not None:
            return np.clip(m, 0.0, 1.0), "vision"
        m = _rembg_mask_for_rgb(rgb_frame)
        return np.clip(m, 0.0, 1.0), "rembg"

    if backend_hint == "rembg":
        m = _rembg_mask_for_rgb(rgb_frame)
        return np.clip(m, 0.0, 1.0), "rembg"

    # auto
    m = _vision_mask_for_rgb(rgb_frame)
    if m is not None:
        return np.clip(m, 0.0, 1.0), "vision"
    m = _rembg_mask_for_rgb(rgb_frame)
    return np.clip(m, 0.0, 1.0), "rembg"

def _process_frame(
    frame_bgr: np.ndarray,
    make_silhouette: bool,
    sil_color: Optional[str],
    bg_fill: Optional[str],
    thresh: Optional[float],
    for_video: bool,
    backend_hint: Optional[str] = None,
):
    """
    Input: BGR uint8 (OpenCV). Returns:
      - if for_video: RGB uint8
      - if for frames: RGBA uint8
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
    h, w, _ = rgb.shape

    mask_soft, _ = _get_soft_mask(rgb, backend_hint=backend_hint)
    if mask_soft.shape[:2] != (h, w):
        mask_soft = cv2.resize(mask_soft, (w, h), interpolation=cv2.INTER_LINEAR)

    m = (mask_soft >= float(thresh)).astype(np.float32) if (thresh is not None) else mask_soft
    m3 = m[..., None]

    if make_silhouette:
        col = _parse_color(sil_color) if sil_color else (0.0, 0.0, 0.0)
        fg = (np.array(col)[None, None, :] * 255.0).astype(np.float32)
        fg_img = np.tile(fg, (h, w, 1))
    else:
        fg_img = rgb.astype(np.float32)

    if for_video:
        if bg_fill is None:
            bg_fill = "black"
        br, bg_, bb = _parse_color(bg_fill)
        bg = (np.array([br, bg_, bb]) * 255.0).astype(np.float32)
        bg_img = np.tile(bg[None, None, :], (h, w, 1))
        comp = fg_img * m3 + bg_img * (1.0 - m3)
        out_rgb = comp.astype(np.uint8)
        return out_rgb
    else:
        alpha = (m * 255.0).astype(np.uint8)
        out_rgba = np.dstack([fg_img.astype(np.uint8), alpha])
        return out_rgba


# ----------------------- audio helpers (ffmpeg) -----------------------
def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def _copy_audio_with_ffmpeg(src_video: str, dst_video: str, verbose: bool = False):
    if not _has_ffmpeg():
        if verbose:
            print("[audio] ffmpeg not found; skipping audio mux.", file=sys.stderr)
        return False
    tmp = str(pathlib.Path(dst_video).with_suffix(".tmp_mux" + pathlib.Path(dst_video).suffix))
    cmd = [
        "ffmpeg", "-y",
        "-i", dst_video,
        "-i", src_video,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c", "copy",
        tmp
    ]
    try:
        if verbose:
            print("[audio] Muxing original audio into output via ffmpeg…", file=sys.stderr)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pathlib.Path(dst_video).unlink(missing_ok=True)
        pathlib.Path(tmp).rename(dst_video)
        if verbose:
            print("[audio] Audio mux complete.", file=sys.stderr)
        return True
    except Exception as e:
        if verbose:
            print(f"[audio] Mux failed: {e}. Leaving video without audio.", file=sys.stderr)
        try:
            pathlib.Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass
        return False


# ----------------------- main processing -----------------------
def process_video(
    in_path: str,
    out_path: str,
    frames_out_dir: Optional[str],
    make_silhouette: bool,
    sil_color: Optional[str],
    bg_fill: Optional[str],
    thresh: Optional[float],
    fps_override: Optional[float],
    keep_audio: bool,
    codec: str,
    verbose: bool,
    force_backend: Optional[str],
):
    t0 = time.time()
    if verbose:
        print("[init] Opening input:", in_path, file=sys.stderr)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {in_path}")

    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = fps_override if fps_override and fps_override > 0 else in_fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1

    write_frames = frames_out_dir is not None or (not out_path or pathlib.Path(out_path).suffix.lower() == "")
    if frames_out_dir is None and write_frames:
        frames_out_dir = out_path if out_path else "frames_out"
    frames_dir = pathlib.Path(frames_out_dir) if write_frames else None
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if not write_frames:
        if verbose:
            print(f"[init] Creating output video: {out_path} ({w}x{h} @ {fps:.3f} fps, codec={codec})", file=sys.stderr)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h), isColor=True)
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create output video: {out_path}")

    # Peek the first frame to determine backend and show info
    ok, first_bgr = cap.read()
    if not ok:
        raise RuntimeError("No frames in input video.")
    # determine backend once using first frame (for logging)
    _, backend_used = _get_soft_mask(cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB), backend_hint=force_backend)
    if verbose:
        mode = "FRAMES (PNG, alpha)" if write_frames else "VIDEO (composited)"
        print(f"[info] Mode: {mode}", file=sys.stderr)
        print(f"[info] Backend: {backend_used} (forced={force_backend or 'auto'})", file=sys.stderr)
        print(f"[info] Input: {w}x{h}, fps={in_fps:.3f}, frames≈{frame_count if frame_count>=0 else 'unknown'}", file=sys.stderr)
        print(f"[info] Options: silhouette={make_silhouette}, sil_color={sil_color or 'black'}, "
              f"bg={bg_fill or ('transparent' if write_frames else 'black')}, "
              f"thresh={'soft' if thresh is None else thresh}", file=sys.stderr)

    # Reset to frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    iterator = range(frame_count) if frame_count > 0 else iter(int, 1)  # endless if unknown
    bar = None
    if tqdm is not None and frame_count > 0:
        bar = tqdm(total=frame_count, unit="f", desc="Processing", leave=True)

    idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            out_for_video = not write_frames
            processed = _process_frame(
                frame_bgr, make_silhouette, sil_color, bg_fill, thresh,
                for_video=out_for_video, backend_hint=force_backend or backend_used
            )

            if write_frames:
                if processed.shape[2] == 3:
                    rgba = np.dstack([processed, np.full(processed.shape[:2], 255, np.uint8)])
                else:
                    rgba = processed
                Image.fromarray(rgba).save(frames_dir / f"frame_{idx:06d}.png")
            else:
                out_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                writer.write(out_bgr)

            idx += 1
            if bar:
                bar.update(1)
            elif verbose and (idx % 30 == 0):
                print(f"[progress] processed {idx} frames…", file=sys.stderr)
    finally:
        cap.release()
        if writer:
            writer.release()
        if bar:
            bar.close()

    if (not write_frames) and keep_audio:
        _copy_audio_with_ffmpeg(in_path, out_path, verbose=verbose)

    if verbose:
        dt = time.time() - t0
        fps_eff = idx / dt if dt > 0 else 0.0
        print(f"[done] Frames: {idx} | elapsed: {dt:.2f}s | ~{fps_eff:.2f} fps", file=sys.stderr)


# ----------------------- CLI -----------------------
def main():
    ap = argparse.ArgumentParser(
        description="Video background removal or silhouettes with progress. "
                    "Uses macOS Vision when available; falls back to rembg."
    )
    ap.add_argument("input", help="Input video file")
    ap.add_argument("-o", "--out", required=True,
                    help="Output video file (e.g., out.mp4) OR output frames directory when --frames is used.")
    ap.add_argument("--frames", action="store_true",
                    help="Write processed frames (PNG) into the -o directory instead of a video.")
    ap.add_argument("-s", "--sil", action="store_true",
                    help="Make a silhouette instead of keeping colors.")
    ap.add_argument("-c", "--sil-color", default=None,
                    help="Silhouette color (e.g., 'black', '#ff00ff', '255,0,0'). Default: black.")
    ap.add_argument("-b", "--bg", default=None,
                    help="Background fill color. "
                         "Frames mode: default transparent. Video mode: default black. "
                         "Accepts 'white','black','#rrggbb' or 'r,g,b' (0–1 or 0–255).")
    ap.add_argument("-t", "--thresh", type=float, default=None,
                    help="Optional mask threshold in [0..1]. If omitted, uses SOFT mask.")
    ap.add_argument("--fps", type=float, default=None,
                    help="Override output video FPS.")
    ap.add_argument("--keep-audio", action="store_true",
                    help="Try to keep original audio (requires ffmpeg).")
    ap.add_argument("--codec", default="mp4v",
                    help="FourCC video codec for OpenCV writer (default: mp4v). Examples: mp4v, avc1, H264 (if available).")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Print detailed progress / backend info.")
    ap.add_argument("--backend", choices=["auto", "vision", "rembg"], default="auto",
                    help="Force a specific mask backend (default: auto).")
    args = ap.parse_args()

    in_path = args.input
    out_path = args.out
    frames_dir = out_path if args.frames or pathlib.Path(out_path).suffix == "" else None

    process_video(
        in_path=in_path,
        out_path=out_path,
        frames_out_dir=frames_dir if args.frames else None,
        make_silhouette=args.sil,
        sil_color=args.sil_color,
        bg_fill=args.bg,
        thresh=args.thresh,
        fps_override=args.fps,
        keep_audio=args.keep_audio,
        codec=args.codec,
        verbose=args.verbose,
        force_backend=(None if args.backend == "auto" else args.backend),
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
