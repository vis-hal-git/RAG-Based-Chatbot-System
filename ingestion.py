# ingestion.py
from pathlib import Path
import pdfplumber
from PIL import Image
import pytesseract
import io
import os
import base64
import uuid

# optionally set tesseract cmd from env
if os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")

# directory to save extracted images
IMG_STORE_DIR = os.path.join(os.getcwd(), "extracted_images")
os.makedirs(IMG_STORE_DIR, exist_ok=True)


def _save_pil_image(img: Image.Image, prefix="img"):
    """Save PIL image to extracted_images and return file path."""
    fn = f"{prefix}_{uuid.uuid4().hex}.png"
    path = os.path.join(IMG_STORE_DIR, fn)
    img.save(path, format="PNG")
    return path


def extract_text_from_pdf(path: str):
    """
    Return list of {'content': str, 'meta': {...}} for text and OCRed images in PDF,
    and also returns image-documents for visual content.
    """
    results = []          # text / OCR text docs
    image_results = []    # image docs with file path in meta

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            meta = {"source": str(path), "page": i + 1, "type": "text"}
            if text.strip():
                results.append({"content": text, "meta": meta})

            # Extract images (as crops) and do OCR for any text inside them
            try:
                if page.images:
                    page_img = page.to_image(resolution=300).original  # higher res for graphs
                    for img_idx, img in enumerate(page.images):
                        try:
                            bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                            crop = page_img.crop(bbox)
                        except Exception:
                            # fallback: save the whole page image for safety
                            crop = page_img

                        # Save image to disk so we can pass it to vision LLM later
                        saved_path = _save_pil_image(crop, prefix=f"page{i+1}_img{img_idx}")

                        # Try OCR on the crop — keep text if exists
                        try:
                            ocr_text = pytesseract.image_to_string(crop)
                        except Exception:
                            ocr_text = ""

                        if ocr_text.strip():
                            results.append({
                                "content": ocr_text,
                                "meta": {"source": str(path), "page": i + 1, "image_index": img_idx, "type": "image_ocr"}
                            })

                        # Also return the image doc (no "content" text, but path in meta)
                        image_results.append({
                            "content": "",  # empty content - image stored in meta
                            "meta": {"source": str(path), "page": i + 1, "image_index": img_idx, "type": "image", "image_path": saved_path}
                        })
            except Exception:
                # ignore image extraction errors for robustness
                pass

    # combine: text first, images after (so indexing order is consistent)
    return results + image_results


def extract_text_from_image(path: str):
    """OCR an image file and also return the saved image path as a separate image-doc."""
    img = Image.open(path)
    text = pytesseract.image_to_string(img) or ""
    # save a copy in extracted_images for consistent handling
    saved_path = _save_pil_image(img, prefix="uploaded_img")
    docs = []
    if text.strip():
        docs.append({"content": text, "meta": {"source": str(path), "type": "image_ocr"}})
    # always include the image doc so the vision LLM can receive it
    docs.append({"content": "", "meta": {"source": str(path), "type": "image", "image_path": saved_path}})
    return docs


def ingest(path: str):
    """Detect type and return list of docs with content + meta (text docs and image docs)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    elif suffix in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        return extract_text_from_image(path)
    else:
        # treat as plain text
        text = p.read_text(encoding='utf8', errors='ignore')
        return [{"content": text, "meta": {"source": str(path), "type": "text"}}]
