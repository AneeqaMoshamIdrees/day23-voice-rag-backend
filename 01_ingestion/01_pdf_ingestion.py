"""
================================================================================
 DAY 17 - STAGE 1: PDF -> STRUCTURED MARKDOWN (Docling-based)
================================================================================
WHAT THIS FILE DOES:
  Reads your PDF from data/ and produces ONE clean file: output/structured.md

  It keeps headings, paragraphs, tables (as markdown tables), formulas
  (as LaTeX), and pictures/diagrams/flowcharts (saved as real .png files
  and linked into the markdown).

  It ALSO embeds invisible page markers like:  <!-- page:12 -->
  These don't show up when you read the file normally, but they let
  Stage 2 (chunking) know exactly which PDF page each piece of text
  came from. Without this, we'd lose page numbers once everything is
  merged into one big markdown file.

PROJECT LAYOUT THIS SCRIPT ASSUMES (run it from the project root):
  day17_embeddings_vector_db/
    data/
      MACHINE LEARNING.pdf
    ingestion/
      ingestion_pipeline.py   <- this file
    output/                    <- created automatically
      structured.md
      images/

HOW TO RUN (from project root, with venv17 active):
  pip install docling --break-system-packages
  python ingestion/ingestion_pipeline.py
================================================================================
"""

import re
import time
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import (
    TextItem,
    SectionHeaderItem,
    TableItem,
    PictureItem,
    DocItemLabel,
)

# ------------------------------------------------------------------------
# CONFIG - paths are relative to the PROJECT ROOT, not to this file.
# Always run this script from the project root folder.
# ------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # .../day17_embeddings_vector_db
PDF_PATH = PROJECT_ROOT / "data" / "MACHINE LEARNING.pdf"   # adjust filename if needed
OUTPUT_DIR = PROJECT_ROOT / "data"          # structured.md + images/ live next to the PDF
IMAGES_DIR = OUTPUT_DIR / "images"
MARKDOWN_PATH = OUTPUT_DIR / "structured.md"


def setup_output_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)
    print(f"[setup] Output folder ready at: {OUTPUT_DIR}")


def convert_pdf(pdf_path: Path):
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found at {pdf_path}. Check the exact filename in data/ "
            f"and update PDF_PATH at the top of this script if it differs."
        )

    print(f"[convert] Reading PDF: {pdf_path.name} (first run can take a while)...")
    start = time.time()

    # By default Docling only detects WHERE pictures are (bounding boxes)
    # but doesn't render the actual image crop. We need generate_picture_images=True
    # to get real .png data back from picture_item.get_image(doc) later.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2.0   # higher = sharper saved images

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(str(pdf_path))
    doc = result.document

    elapsed = time.time() - start
    print(f"[convert] Done in {elapsed:.1f}s. "
          f"Pages: {len(doc.pages)} | Texts: {len(doc.texts)} | "
          f"Tables: {len(doc.tables)} | Pictures: {len(doc.pictures)}")
    return doc


def table_to_markdown(table_item: TableItem, doc) -> str:
    """Turns a Docling table into a clean Markdown table string."""
    try:
        df = table_item.export_to_dataframe(doc)
    except Exception as e:
        return f"*[Table could not be parsed: {e}]*"

    if df.empty:
        return "*[Empty table]*"

    headers = list(df.columns.astype(str))
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = [
        "| " + " | ".join(str(v) for v in row.tolist()) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([header_row, separator_row] + body_rows)


def save_picture(picture_item: PictureItem, doc, image_index: int, prefix: str = "") -> str:
    """
    Saves a picture/diagram/flowchart as a real .png. Returns its relative path.

    `prefix` is used to keep filenames unique ACROSS different documents --
    without it, uploading a second PDF would produce the same
    "figure_001.png", "figure_002.png", ... names as the first document
    and silently overwrite its images.
    """
    image_filename = f"{prefix}figure_{image_index:03d}.png"
    image_path = IMAGES_DIR / image_filename
    try:
        pil_image = picture_item.get_image(doc)
        if pil_image is not None:
            pil_image.save(image_path)
            return f"images/{image_filename}"
        else:
            print(f"[warn] Picture {image_index} on page "
                  f"{picture_item.prov[0].page_no if picture_item.prov else '?'} "
                  f"had no image data (get_image returned None)")
    except Exception as e:
        print(f"[warn] Could not save image {image_index}: {e}")
    return ""


def is_formula(text_item: TextItem) -> bool:
    return getattr(text_item, "label", None) == DocItemLabel.FORMULA


def build_markdown(doc, image_prefix: str = "") -> str:
    """
    Walks the document in reading order and builds one markdown string.
    Inserts a hidden <!-- page:N --> comment every time the page number
    changes, so Stage 2 can recover page numbers per chunk.

    `image_prefix` is forwarded to save_picture() to keep filenames
    unique when this function is called for more than one document
    (see ingest_pdf_to_markdown below).
    """
    parts = []
    last_page = None
    image_index = 0

    for item, _level in doc.iterate_items():
        page_no = item.prov[0].page_no if item.prov else last_page

        if page_no is not None and page_no != last_page:
            parts.append(f"\n<!-- page:{page_no} -->\n")
            last_page = page_no

        if isinstance(item, SectionHeaderItem):
            parts.append(f"\n## {item.text.strip()}\n")

        elif isinstance(item, TextItem) and is_formula(item):
            parts.append(f"\n$$\n{item.text.strip()}\n$$\n")

        elif isinstance(item, TextItem):
            text = item.text.strip()
            if text:
                parts.append(text)

        elif isinstance(item, TableItem):
            parts.append(f"\n{table_to_markdown(item, doc)}\n")

        elif isinstance(item, PictureItem):
            image_index += 1
            rel_path = save_picture(item, doc, image_index, prefix=image_prefix)
            caption = ""
            try:
                caption = item.caption_text(doc) or ""
            except Exception:
                pass
            if rel_path:
                parts.append(f"\n![{caption or 'figure'}]({rel_path})\n")
                if caption:
                    parts.append(f"*Figure: {caption}*\n")

    return "\n".join(parts)


def save_markdown(markdown_body: str, title="Machine Learning - Lecture Notes"):
    full_markdown = f"# {title}\n\n" + markdown_body
    MARKDOWN_PATH.write_text(full_markdown, encoding="utf-8")
    print(f"[save] Structured markdown written to: {MARKDOWN_PATH}")


def main():
    setup_output_dirs()
    doc = convert_pdf(PDF_PATH)
    markdown_body = build_markdown(doc)
    save_markdown(markdown_body)
    print("\n[done] Stage 1 complete. Next: run chunking/recursive_chunker.py")


if __name__ == "__main__":
    main()


def _safe_filename_prefix(pdf_path: Path) -> str:
    """
    Turns a PDF filename into a short, filesystem-safe prefix, e.g.
    "Comparison with Existing Solutions.pdf" -> "Comparison_with_Existing_Solutions_"
    """
    stem = re.sub(r"[^a-zA-Z0-9]+", "_", pdf_path.stem).strip("_")
    return f"{stem}_" if stem else "doc_"


def ingest_pdf_to_markdown(pdf_path: Path) -> str:
    """
    Runs the same Stage 1 pipeline as main(), but on an arbitrary PDF path
    and returns the markdown as a string instead of writing to the shared
    data/structured.md — so it never clobbers the default document's file.

    Images are saved with a filename prefix derived from the PDF's own
    name, so uploading multiple documents never overwrites another
    document's figures.

    Used by /api/rag/ingest for user-uploaded documents.
    """
    setup_output_dirs()
    doc = convert_pdf(pdf_path)
    prefix = _safe_filename_prefix(pdf_path)
    markdown_body = build_markdown(doc, image_prefix=prefix)
    title = pdf_path.stem
    return f"# {title}\n\n" + markdown_body