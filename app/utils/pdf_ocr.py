"""
PDF text extraction and validation utilities.

Uses PyMuPDF (fitz) for PDF parsing/rendering and pytesseract for OCR
fallback on scanned/image-only pages or standalone images.
"""

import base64
import io
import re

import pymupdf as fitz 

# Matches a hyphen at end-of-line, tolerating stray trailing spaces/tabs
# before the newline (common in justified-text PDF extraction), followed
# by a lowercase letter continuing the word on the next line, e.g.
# "informa-\ntion" or "informa- \ntion" -> "information"
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w+)-[ \t]*\n[ \t]*([a-z]\w*)")

# Prefixes/suffixes where the hyphen is almost always a genuine compound
# separator, not a line-wrap artifact -- never merge these, even if they
# happen to fall at a line break.
_HYPHEN_KEEP_PARTS = frozenset(
    {
        "non", "pre", "re", "self", "well", "co", "sub", "multi",
        "anti", "post", "semi", "mid", "cross", "inter", "over",
        "under", "ex", "quasi", "pseudo", "counter",
    }
)

# Small set of common ligatures / typographic characters that OCR and PDF
# text extraction frequently emit. Fixed via a direct translation table
# instead of full Unicode NFKC normalization, which also collapses
# semantically meaningful characters (superscripts, No/№, Roman numeral
# compatibility forms, etc.) that we don't want to silently alter.
_LIGATURE_MAP = str.maketrans(
    {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "st",
        "\ufb06": "st",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",  # en dash
        "\u2014": "--",  # em dash
        "\u00a0": " ",  # non-breaking space
    }
)

# Collapses 3+ blank lines down to a single paragraph break
_MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")

# Collapses runs of horizontal whitespace (spaces/tabs) to a single space
_HORIZONTAL_WS_RE = re.compile(r"[ \t]+")

# Strips trailing whitespace at the end of each line
_TRAILING_WS_RE = re.compile(r"[ \t]+(?=\n)")

# Matches a line containing only whitespace (so it can be treated as blank)
_WHITESPACE_ONLY_LINE_RE = re.compile(r"^[ \t]+$", re.MULTILINE)


def _dehyphenate(match: re.Match) -> str:
    left, right = match.group(1), match.group(2)

    if left.lower() in _HYPHEN_KEEP_PARTS:
        return f"{left}-{right}"

    return f"{left}{right}"


def normalize_text(text: str) -> str:
    """
    Normalize extracted PDF/OCR text for consistent downstream processing.

    - Fixes common ligatures and typographic characters (e.g. "ﬁ" -> "fi",
      smart quotes, en/em dashes) via a targeted translation table, without
      the side effects of full Unicode NFKC normalization
    - Strips trailing whitespace from each line first, so hyphen detection
      isn't thrown off by stray spaces before a line break
    - Rejoins words hyphenated across a line break ("informa-\\ntion" ->
      "information"), but leaves the hyphen in place for common compound
      prefixes/suffixes ("well-\\nbeing" -> "well-being", not "wellbeing")
    - Treats whitespace-only lines as blank before collapsing blank runs
    - Collapses runs of spaces/tabs to a single space
    - Collapses 3+ consecutive blank lines down to one blank line
    - Strips leading/trailing whitespace from the whole string

    Args:
        text: Raw extracted or OCR'd text.

    Returns:
        str: Normalized text.
    """
    if not text:
        return text

    text = text.translate(_LIGATURE_MAP)
    text = _TRAILING_WS_RE.sub("", text)
    text = _HYPHEN_LINEBREAK_RE.sub(_dehyphenate, text)
    text = _HORIZONTAL_WS_RE.sub(" ", text)
    text = _WHITESPACE_ONLY_LINE_RE.sub("", text)
    text = _MULTI_BLANK_LINE_RE.sub("\n\n", text)

    return text.strip()


def validate_image_bytes(contents: bytes) -> None:
    """
    Verify that `contents` decodes as a valid image.

    Raises:
        ValueError: If the bytes do not decode as a valid image.
    """
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(contents)) as img:
            img.verify()
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"File content does not match a valid image: {e}")


def split_pdf_bytes(pdf_bytes: bytes):
    """
    Split a PDF into individual single-page PDFs.

    Args:
        pdf_bytes: Raw PDF file content.

    Yields:
        tuple[int, bytes]: (1-indexed page number, single-page PDF bytes)
                           for each page in the source document.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    try:
        for i in range(doc.page_count):
            single_page_doc = fitz.open()

            try:
                single_page_doc.insert_pdf(
                    doc,
                    from_page=i,
                    to_page=i,
                )

                yield i + 1, single_page_doc.tobytes()

            finally:
                single_page_doc.close()

    finally:
        doc.close()


def render_pdf_images(
    pdf_bytes: bytes,
    dpi: int = 300,
    mode: str = "b64",
):
    """
    Render each page of a PDF to an image.

    Args:
        pdf_bytes: Raw PDF file content.
        dpi: Render resolution.
        mode:
            - "b64": Base64-encoded PNG string.
            - "bytes": Raw PNG bytes.
            - "image": PIL.Image.Image object.

    Yields:
        tuple[int, str | bytes | PIL.Image.Image]:
            (1-indexed page number, rendered page).

    Raises:
        ValueError: If `mode` is invalid.
    """
    if mode not in ("b64", "bytes", "image"):
        raise ValueError(
            f"Invalid mode: {mode!r}. "
            "Must be 'b64', 'bytes', or 'image'."
        )

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    try:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi)
            png_bytes = pix.tobytes("png")

            if mode == "bytes":
                yield i + 1, png_bytes

            elif mode == "b64":
                yield (
                    i + 1,
                    base64.b64encode(png_bytes).decode("ascii"),
                )

            else:
                from PIL import Image

                yield i + 1, Image.open(io.BytesIO(png_bytes))

    finally:
        doc.close()


def validate_pdf_bytes(
    contents: bytes,
    reject_encrypted: bool = True,
) -> None:
    """
    Verify that `contents` decodes as a valid PDF with at least one page.

    Args:
        contents: Raw PDF file content.
        reject_encrypted: Whether encrypted PDFs should be rejected.

    Raises:
        ValueError: If the bytes do not form a valid PDF, contain no
                    pages, or are encrypted when rejection is enabled.
    """
    try:
        doc = fitz.open(
            stream=contents,
            filetype="pdf",
        )
    except Exception as e:
        raise ValueError(
            f"File content does not match a valid PDF: {e}"
        )

    try:
        if doc.page_count < 1:
            raise ValueError("PDF contains no pages")

        if reject_encrypted and doc.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported")

    finally:
        doc.close()


def extract_pdf_text(
    pdf_bytes: bytes,
    per_page: bool = False,
    ocr_fallback: bool = False,
    layout: bool = False,
    normalize: bool = True,
):
    """
    Extract text from PDF bytes using PyMuPDF.

    Args:
        pdf_bytes: Raw PDF file content.
        per_page:
            If True, return one string per page.
            If False, return one concatenated string.
        ocr_fallback:
            If True, OCR pages that contain no extractable PDF text.
        layout:
            If True, preserve approximate reading order using text
            block positions.
        normalize:
            If True, run extracted text through normalize_text()
            (Unicode NFKC, dehyphenation, whitespace cleanup).

    Returns:
        str | list[str]:
            Extracted text, either concatenated or per-page.
    """
    pages_text = []

    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf",
    )

    try:
        for page in doc:
            if layout:
                text = _extract_layout_text(page)
            else:
                text = page.get_text("text")

            if not text.strip() and ocr_fallback:
                text = _ocr_page(page)

            if normalize:
                text = normalize_text(text)

            pages_text.append(text)

    finally:
        doc.close()

    return pages_text if per_page else "\n\n".join(pages_text)


def extract_image_text(image_bytes: bytes, normalize: bool = True) -> str:
    """
    OCR a standalone image given as raw bytes.

    Args:
        image_bytes: Raw JPEG/PNG/etc. image bytes.
        normalize: If True, run OCR output through normalize_text().

    Returns:
        str: OCR-extracted text.
    """
    import pytesseract
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))

    try:
        text = pytesseract.image_to_string(img)
        return normalize_text(text) if normalize else text
    finally:
        img.close()


def _extract_layout_text(page) -> str:
    """
    Extract text while preserving approximate reading order
    using block positions.
    """
    blocks = page.get_text("blocks")

    blocks = sorted(
        blocks,
        key=lambda b: (
            round(b[1], 1),
            b[0],
        ),
    )

    return "\n".join(
        block[4]
        for block in blocks
        if block[4].strip()
    )


def _ocr_page(
    page,
    dpi: int = 300,
) -> str:
    """
    OCR a single PDF page as a fallback for scanned/image-only pages.
    """
    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(
        io.BytesIO(pix.tobytes("png"))
    )

    try:
        return pytesseract.image_to_string(img)
    finally:
        img.close()