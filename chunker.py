# chunker.py
import re

DEFAULT_CHUNK_CHARS = 1200  # rough charter-to-token proxy


def paragraph_split(text):
    # split by empty lines or long newlines
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not parts:
        # fallback: split by sentence-like punctuation
        return [text[i:i+DEFAULT_CHUNK_CHARS] for i in range(0, len(text), DEFAULT_CHUNK_CHARS)]
    return parts


def chunk_document(content: str, meta: dict, chunk_chars: int = DEFAULT_CHUNK_CHARS):
    chunks = []
    paras = paragraph_split(content)
    current = ""
    for p in paras:
        if len(current) + len(p) <= chunk_chars:
            current = (current + "\n\n" + p).strip()
        else:
            if current:
                chunks.append({"content": current, "meta": dict(meta)})
            # if p itself > chunk size, break it
            if len(p) > chunk_chars:
                for i in range(0, len(p), chunk_chars):
                    sub = p[i:i+chunk_chars]
                    chunks.append({"content": sub, "meta": dict(meta)})
                current = ""
            else:
                current = p
    if current:
        chunks.append({"content": current, "meta": dict(meta)})
    # attach snippet to meta for quick preview
    for c in chunks:
        c["meta"].setdefault("snippet", c["content"][:400])
    return chunks
