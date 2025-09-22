# Silhouette Maker (macOS-first, cross-platform fallback)

Turn photos into **black silhouettes**.
On macOS 12+ the script first tries Apple’s **Vision** framework for high-quality subject/person masks.
If Vision isn’t available (or fails), it **automatically falls back** to a pure-Python pipeline using **rembg + onnxruntime**.

## What it does

![pp2](https://github.com/user-attachments/assets/4e37546c-119b-4c7b-b7d3-78e440287968)

<img width="994" height="559" alt="pp2-nb" src="https://github.com/user-attachments/assets/8528e3d2-e3ab-47bd-8fc3-56cb1041264a" />
<img width="994" height="559" alt="pp2-br" src="https://github.com/user-attachments/assets/6b15ead7-e104-4e63-a119-eb6ac330e1ed" />
<img width="994" height="559" alt="pp2-br-white" src="https://github.com/user-attachments/assets/a461aff2-71ea-4063-a8b3-991c5c2ed53c" />


* Takes one or more images and produces **black silhouettes**:

  * Transparent background (PNG with alpha), or
  * Solid background color (e.g., white), if you pass `--bg`.
* Uses:

  * **macOS Vision** (preferred):

    * macOS 14+: general foreground instance masks (any subject)
    * macOS 12–13: person segmentation (people only)
  * **Fallback** (all platforms): `rembg` (U²-Net) via `onnxruntime`

---

## Install

> Python ≥ 3.10 recommended.
> These steps show a clean virtual environment setup.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
```

### A) macOS with Apple Vision (fastest, highest quality on Apple devices)

Install PyObjC bindings for Vision & Quartz:

```bash
pip install "pyobjc>=10" pyobjc-framework-Vision pyobjc-framework-Quartz
```

Also install the fallback (so it can still run outside Vision’s capabilities):

```bash
pip install rembg onnxruntime pillow numpy
```

> On Apple Silicon, `onnxruntime` universal2 wheels usually work. If you hit runtime errors, try:
>
> ```bash
> pip install onnxruntime-silicon
> ```

### B) Cross-platform (no macOS Vision)

Just install the fallback stack:

```bash
pip install rembg onnxruntime pillow numpy
```

---

## Usage

Place the script (e.g., `silhouette_mac.py`) at the repo root.

### Single image → PNG with transparency

```bash
python silhouette_mac.py path/to/input.jpg out.png
```

### Multiple images → write to a folder

```bash
python silhouette_mac.py img1.jpg img2.png img3.jpeg out_dir/
```

This creates files like `out_dir/img1_silhouette.png`.

### Solid background (e.g., white)

```bash
python silhouette_mac.py input.jpg out.png --bg white
# also supports hex or RGB:
# --bg "#00aaff"     (hex)
# --bg "34,139,34"   (RGB 0–255)
# --bg "0.13,0.5,1"  (RGB 0–1 floats)
```

### Control mask threshold (0..1, default 0.5)

Lower threshold → more area becomes foreground; higher → tighter silhouettes.

```bash
python silhouette_mac.py input.jpg out.png --threshold 0.6
```

---

## How it works (brief)

1. **Vision path (macOS)**

   * Tries `VNGenerateForegroundInstanceMaskRequest` (macOS 14+) for general subjects.
   * If unavailable, tries `VNGeneratePersonSegmentationRequest` (macOS 12–13).

2. **Fallback path (any OS)**

   * Uses `rembg` to remove background and extracts the alpha channel as a mask.

3. **Composition**

   * Binarizes the soft mask at `--threshold`.
   * Produces a black foreground:

     * PNG with **transparent** background (default), or
     * **Solid** background color if `--bg` is provided.

---

## Troubleshooting

* **`ModuleNotFoundError: No module named 'rembg'`**
  Install it:

  ```bash
  pip install rembg onnxruntime
  ```

* **Apple Vision path fails / unavailable**
  Ensure PyObjC bits are installed:

  ```bash
  pip install "pyobjc>=10" pyobjc-framework-Vision pyobjc-framework-Quartz
  ```

  The script will still run via the fallback if Vision can’t be used.

* **Apple Silicon ONNX runtime issues**
  Try the silicon build:

  ```bash
  pip install onnxruntime-silicon
  ```

* **Pillow deprecation warning about `mode=`**
  Safe to ignore; will be removed in a future Pillow major version. Doesn’t affect output.

---

## Examples

```bash
# Transparent silhouette
python silhouette_mac.py images/pp2.jpg images/pp2-nb.png

# Batch, white background
python silhouette_mac.py photos/*.jpg out/ --bg white

# Tighter mask
python silhouette_mac.py input.png out.png --threshold 0.65
```

---

## Roadmap

* [ ] **Binary mask output option** (`--mask-out` to save the mask as a grayscale PNG)
* [ ] **Color fill option** (e.g., non-black silhouettes: `--fill "#ff00ff"`)
* [ ] **Edge smoothing / feathering** (`--feather px`) to reduce jaggies
* [ ] **Batch parallelism** for large folders
* [ ] **Background blur** (when using solid color isn’t desired)
* [ ] **Model choice for fallback** (e.g., different rembg models)
* [ ] **Homebrew wrapper / CLI tool** (`pipx`/`brew` formula for 1-line install)
* [ ] **Simple GUI** (drag-and-drop)

---

## License

Your choice (e.g., MIT). Add a `LICENSE` file if you want to distribute.
