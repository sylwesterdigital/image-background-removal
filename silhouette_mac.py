#!/usr/bin/env python3
"""
Silhouette maker for macOS that prefers Apple's Vision (if available) and
falls back to pure-Python (rembg + onnxruntime) automatically.

Usage:
  python silhouette.py input.jpg out.png
  python silhouette.py img1.jpg img2.png out_dir/ --bg white --threshold 0.6
"""

import argparse, pathlib, sys, ctypes
from typing import Optional, Tuple

import numpy as np
from PIL import Image

# ------------------------------------------------------------
# Optional macOS Vision path (works on macOS 12+ with PyObjC)
# ------------------------------------------------------------
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

def _vision_mask(path: str) -> Optional[np.ndarray]:
    """Return soft mask in [0..1] or None if Vision path is unavailable."""
    imported = _try_import_vision()
    if not imported:
        return None
    VN, Q, NSURL, CV = imported

    try:
        cg = _load_cgimage(Q, NSURL, path)
        w, h = _cgsize(Q, cg)

        # Create handler WITHOUT calling a potentially NS_UNAVAILABLE init
        # Use class constructor that exists in PyObjC:
        # VNImageRequestHandler.imageRequestHandlerWithCGImage:options_
        handler = VN.VNImageRequestHandler.imageRequestHandlerWithCGImage_options_(cg, {})

        # Try modern foreground instance mask (macOS 14+)
        try:
            if hasattr(VN, "VNGenerateForegroundInstanceMaskRequest"):
                # Avoid init/new; construct via alloc().initWith… if available,
                # otherwise try convenience constructor `.request()`.
                req = None
                for ctor in (
                    lambda: VN.VNGenerateForegroundInstanceMaskRequest.alloc().initWithRevision_error_(1, None),
                    lambda: VN.VNGenerateForegroundInstanceMaskRequest.request(),  # may exist
                    lambda: VN.VNGenerateForegroundInstanceMaskRequest.alloc().init(),  # last resort
                ):
                    try:
                        req = ctor()
                        if req is not None:
                            break
                    except Exception:
                        req = None
                if req is not None:
                    ok = handler.performRequests_error_([req], None)
                    if ok and req.results() and len(req.results()) > 0:
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
            # 8-bit, one channel
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

# ------------------------------------------------------------
# Pure-Python fallback via rembg (works cross-platform)
# ------------------------------------------------------------
def _rembg_mask(path: str) -> np.ndarray:
    """
    Returns a soft alpha mask [0..1] using rembg.
    On Apple Silicon, ensure 'onnxruntime-silicon' is installed.
    """
    import io
    from rembg import remove
    with open(path, "rb") as f:
        data = f.read()
    rgba = Image.open(io.BytesIO(remove(data))).convert("RGBA")
    alpha = np.array(rgba)[:, :, 3].astype(np.float32) / 255.0
    return alpha

# ------------------------------------------------------------
# Composition helpers
# ------------------------------------------------------------
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

def _write_silhouette_for_image(inp: str, out: str, bg: Optional[str], threshold: float):
    # 1) get a soft mask
    mask = _vision_mask(inp)
    if mask is None:
        mask = _rembg_mask(inp)

    # 2) binarize for crisp silhouette
    mask_bin = (mask >= float(threshold)).astype(np.uint8)

    # 3) compose
    h, w = mask_bin.shape
    alpha = (mask_bin * 255).astype(np.uint8)
    fg_rgb = np.zeros((h, w, 3), dtype=np.uint8)  # black silhouette

    if bg is None:
        rgba = np.dstack([fg_rgb, alpha])  # transparent bg
        Image.fromarray(rgba, mode="RGBA").save(out)
    else:
        br, bg_, bb = _parse_color(bg)
        bg_img = (np.array([br, bg_, bb]) * 255.0).astype(np.uint8)
        bg_rgb = np.tile(bg_img[None, None, :], (h, w, 1))
        m = mask_bin[..., None].astype(np.uint8)
        # composite: out = fg*m + bg*(1-m); fg is 0 → just bg*(1-m)
        comp = (bg_rgb * (1 - m)).astype(np.uint8)
        Image.fromarray(comp, mode="RGB").save(out)

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Create black silhouettes (Vision on macOS if available, else rembg).")
    ap.add_argument("inputs", nargs="+", help="Input image(s)")
    ap.add_argument("output", help="Output .png OR output directory")
    ap.add_argument("--bg", default=None, help="Background color (default transparent). "
                    "Accepted: 'white','black','#rrggbb' or 'r,g,b' (0–1 or 0–255).")
    ap.add_argument("--threshold", type=float, default=0.5, help="Mask threshold in [0,1]")
    args = ap.parse_args()

    out = pathlib.Path(args.output)
    many = len(args.inputs) > 1 or out.suffix.lower() != ".png"
    if many:
        out.mkdir(parents=True, exist_ok=True)
        for p in args.inputs:
            dst = out / (pathlib.Path(p).stem + "_silhouette.png")
            _write_silhouette_for_image(p, str(dst), args.bg, args.threshold)
            print("Wrote", dst)
    else:
        _write_silhouette_for_image(args.inputs[0], str(out), args.bg, args.threshold)
        print("Wrote", out)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
