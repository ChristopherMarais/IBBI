# src/ibbi/models/_gpu.py

"""
GPU image handling for the fast inference path.

Profiling the identification pipeline on 12.6-megapixel benchmark photographs (RTX PRO 6000) showed that most of the
time went to CPU work on the full-resolution image: JPEG decoding (31%) and Ultralytics' CPU letterbox (most of the
detector's 45%), while the networks themselves took little. These helpers keep the image on the GPU instead:

* `load_tensor`: decode JPEG files with nvJPEG (torchvision) on the GPU, EXIF orientation applied; any other input is
  loaded with `load_image` and uploaded.
* `ultralytics_letterbox`: the resize-and-pad of Ultralytics' `LetterBox` (rectangular inference, centred, value 114,
  bilinear), done on the GPU, so the detector receives a ready tensor.
* `crop_letterbox`: the classifier's crop (box + padding) and square letterbox (bicubic, antialiased) for many boxes.

Resizing on the GPU is not bit-identical to OpenCV / Pillow (differences of about one intensity level); the benchmark
results of the fast path are checked against the reference path in the test suite and `docs/benchmark.md`.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ._common import ImageInput, load_image

_JPEG = (".jpg", ".jpeg", ".jpe", ".jfif")


def load_tensor(image: ImageInput | torch.Tensor, device: str) -> torch.Tensor:
    """Returns the image as a uint8 RGB tensor [3, H, W] on `device` (EXIF orientation applied)."""
    if isinstance(image, torch.Tensor):
        t = image if image.dim() == 3 else image[0]
        if t.shape[0] != 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1)
        return t.to(device)
    if isinstance(image, (str, Path)) and not str(image).startswith(("http://", "https://")) and Path(image).suffix.lower() in _JPEG:
        import torchvision
        from torchvision.io import ImageReadMode

        data = torchvision.io.read_file(str(image))
        try:
            if str(device).startswith("cuda"):
                t = torchvision.io.decode_jpeg(data, mode=ImageReadMode.RGB, device=device)
                return _exif_orient(t, image)
            return torchvision.io.decode_jpeg(data, mode=ImageReadMode.RGB, apply_exif_orientation=True).to(device)
        except RuntimeError:
            pass  # unusual JPEG (e.g. CMYK, progressive variants nvJPEG rejects): fall back to Pillow
    arr = np.asarray(load_image(image), dtype=np.uint8)
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).to(device)


def _exif_orient(t: torch.Tensor, path) -> torch.Tensor:
    """Applies the EXIF orientation tag like `PIL.ImageOps.exif_transpose`, on a [3, H, W] tensor."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            o = im.getexif().get(0x0112, 1)
    except Exception:
        return t
    if o == 2:
        return t.flip(2)
    if o == 3:
        return t.flip(1).flip(2)
    if o == 4:
        return t.flip(1)
    if o == 5:  # transpose
        return t.transpose(1, 2)
    if o == 6:  # rotate 270 (90 clockwise)
        return t.transpose(1, 2).flip(2)
    if o == 7:  # transverse
        return t.transpose(1, 2).flip(1).flip(2)
    if o == 8:  # rotate 90 counter-clockwise
        return t.transpose(1, 2).flip(1)
    return t


def _resize_uint8(x: torch.Tensor, size: tuple[int, int], mode: str, antialias: bool) -> torch.Tensor:
    """Resizes a uint8 [N, 3, H, W] tensor; returns uint8 (rounded, like OpenCV / Pillow)."""
    y = F.interpolate(x.float(), size=size, mode=mode, align_corners=False, antialias=antialias)
    return y.round_().clamp_(0, 255).to(torch.uint8)


def ultralytics_letterbox(img: torch.Tensor, imgsz: int, stride: int = 32) -> torch.Tensor:
    """Ultralytics `LetterBox(imgsz, auto=True, stride)` for one image, on the GPU.

    Args:
        img: uint8 [3, H, W].
    Returns:
        float [1, 3, h, w] in [0, 1], h and w multiples of `stride` (the input Ultralytics expects as a tensor).
    """
    H, W = img.shape[1:]
    r = min(imgsz / H, imgsz / W)
    nw, nh = round(W * r), round(H * r)
    dw, dh = (imgsz - nw) % stride / 2, (imgsz - nh) % stride / 2
    x = img[None]
    if (W, H) != (nw, nh):
        x = _resize_uint8(x, (nh, nw), "bilinear", antialias=False)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    x = F.pad(x, (left, right, top, bottom), value=114)
    return x.float().div_(255.0)


def crop_letterbox(img: torch.Tensor, boxes, pad: float, res: int, fill: tuple[int, int, int]) -> torch.Tensor:
    """The classifier's crop and letterbox for every box: uint8 [N, 3, res, res].

    Same geometry as `HierarchicalClassifier.crop` (box grown by `pad` on every side, clipped to the image, rounded)
    followed by `_letterbox` (keep ratio, bicubic, centred on a `fill` canvas).
    """
    H, W = img.shape[1:]
    fill_t = torch.tensor(fill, dtype=torch.uint8, device=img.device).view(3, 1, 1)
    out = []
    for b in boxes:
        x0, y0, x1, y1 = (float(v) for v in b)
        w, h = x1 - x0, y1 - y0
        x0, y0, x1, y1 = max(0.0, x0 - pad * w), max(0.0, y0 - pad * h), min(float(W), x1 + pad * w), min(float(H), y1 + pad * h)
        if x1 - x0 < 2 or y1 - y0 < 2:
            c = img
        else:
            c = img[:, int(round(y0)) : int(round(y1)), int(round(x0)) : int(round(x1))]
        ch, cw = c.shape[1:]
        s = res / max(cw, ch)
        nw, nh = max(1, round(cw * s)), max(1, round(ch * s))
        c = _resize_uint8(c[None], (nh, nw), "bicubic", antialias=True)[0]
        canvas = fill_t.expand(3, res, res).clone()
        top, left = (res - nh) // 2, (res - nw) // 2
        canvas[:, top : top + nh, left : left + nw] = c
        out.append(canvas)
    if not out:
        return torch.zeros((0, 3, res, res), dtype=torch.uint8, device=img.device)
    return torch.stack(out)
