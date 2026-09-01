"""
Day23 - Document Element Representation

Converts the structured Markdown produced by the PDF ingestion
stage into clearly identified document elements.

Supported elements:
    - heading
    - text
    - table
    - equation
    - image

The goal is to preserve the structure of the document before
hierarchical chunking.
"""

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

STRUCTURED_MARKDOWN = PROJECT_ROOT / "data" / "structured.md"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "structured_documents"
    / "document_elements.json"
)


def parse_document_elements(markdown_text: str) -> list:
    """Parse structured Markdown into typed document elements."""

    page_pattern = re.compile(r"<!--\s*page:(\d+)\s*-->")
    heading_pattern = re.compile(r"^##\s+(.+)$")
    image_pattern = re.compile(r"^!\[(.*?)\]\((.*?)\)$")
    table_pattern = re.compile(r"^\|.*\|$")
    formula_pattern = re.compile(r"^\$\$$")

    elements = []

    current_page = None
    current_section = "Introduction"

    lines = markdown_text.splitlines()

    text_buffer = []

    def flush_text():
        nonlocal text_buffer

        text = "\n".join(text_buffer).strip()

        if text:
            elements.append(
                {
                    "element_type": "text",
                    "page": current_page,
                    "section": current_section,
                    "content": text,
                }
            )

        text_buffer = []

    i = 0

    while i < len(lines):

        line = lines[i].strip()

        # ---------------------------------------------------------
        # PAGE MARKER
        # ---------------------------------------------------------
        page_match = page_pattern.match(line)

        if page_match:
            flush_text()
            current_page = int(page_match.group(1))
            i += 1
            continue

        # ---------------------------------------------------------
        # HEADING
        # ---------------------------------------------------------
        heading_match = heading_pattern.match(line)

        if heading_match:
            flush_text()

            current_section = heading_match.group(1).strip()

            elements.append(
                {
                    "element_type": "heading",
                    "page": current_page,
                    "section": current_section,
                    "content": current_section,
                }
            )

            i += 1
            continue

        # ---------------------------------------------------------
        # IMAGE
        # ---------------------------------------------------------
        image_match = image_pattern.match(line)

        if image_match:
            flush_text()

            caption = image_match.group(1)
            image_path = image_match.group(2)

            elements.append(
                {
                    "element_type": "image",
                    "page": current_page,
                    "section": current_section,
                    "content": image_path,
                    "caption": caption,
                }
            )

            i += 1
            continue

        # ---------------------------------------------------------
        # TABLE
        # ---------------------------------------------------------
        if table_pattern.match(line):

            flush_text()

            table_lines = []

            while i < len(lines):

                current_line = lines[i].strip()

                if not table_pattern.match(current_line):
                    break

                table_lines.append(current_line)
                i += 1

            elements.append(
                {
                    "element_type": "table",
                    "page": current_page,
                    "section": current_section,
                    "content": "\n".join(table_lines),
                }
            )

            continue

        # ---------------------------------------------------------
        # EQUATION
        # ---------------------------------------------------------
        if formula_pattern.match(line):

            flush_text()

            equation_lines = [line]

            i += 1

            while i < len(lines):

                equation_lines.append(lines[i])

                if lines[i].strip() == "$$":
                    i += 1
                    break

                i += 1

            elements.append(
                {
                    "element_type": "equation",
                    "page": current_page,
                    "section": current_section,
                    "content": "\n".join(equation_lines),
                }
            )

            continue

        # ---------------------------------------------------------
        # NORMAL TEXT
        # ---------------------------------------------------------
        text_buffer.append(lines[i])

        i += 1

    flush_text()

    return elements


def main():

    if not STRUCTURED_MARKDOWN.exists():

        raise FileNotFoundError(
            f"Structured Markdown not found: {STRUCTURED_MARKDOWN}"
        )

    print(f"[load] Reading: {STRUCTURED_MARKDOWN}")

    markdown_text = STRUCTURED_MARKDOWN.read_text(
        encoding="utf-8"
    )

    elements = parse_document_elements(markdown_text)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            elements,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    counts = {}

    for element in elements:

        element_type = element["element_type"]

        counts[element_type] = (
            counts.get(element_type, 0) + 1
        )

    print("\n[done] Document elements created")

    print(f"[save] {OUTPUT_FILE}")

    print("\n[element breakdown]")

    for element_type, count in counts.items():

        print(
            f"  {element_type:10} : {count}"
        )


if __name__ == "__main__":
    main()