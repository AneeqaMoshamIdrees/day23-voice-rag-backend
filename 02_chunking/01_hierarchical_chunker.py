"""
Day23 - Hierarchical Chunking

Creates a two-level document hierarchy:

    Parent
      └── Child chunks

Parent:
    Represents a complete document section.

Children:
    Smaller searchable units belonging to that section.

Special elements such as:
    - tables
    - equations
    - images

are preserved as complete child elements and are never split.
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "structured_documents"
    / "document_elements.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "structured_documents"
    / "hierarchical_chunks.json"
)


# Maximum size for normal text child chunks.
TEXT_CHUNK_SIZE = 800


def split_text(text: str, max_size: int = TEXT_CHUNK_SIZE) -> list[str]:
    """
    Split normal text using meaningful boundaries.

    We prefer:
        paragraphs
        sentences
        words

    before using a hard character split.
    """

    if len(text) <= max_size:
        return [text.strip()]

    # First try paragraphs.
    paragraphs = [
        part.strip()
        for part in text.split("\n\n")
        if part.strip()
    ]

    chunks = []

    for paragraph in paragraphs:

        if len(paragraph) <= max_size:
            chunks.append(paragraph)
            continue

        # Sentence-level splitting.
        sentences = [
            sentence.strip()
            for sentence in paragraph.split(". ")
            if sentence.strip()
        ]

        current = ""

        for sentence in sentences:

            if not current:
                current = sentence
                continue

            candidate = current + ". " + sentence

            if len(candidate) <= max_size:
                current = candidate

            else:
                chunks.append(current)
                current = sentence

        if current:
            chunks.append(current)

    # Last-resort hard split.
    final_chunks = []

    for chunk in chunks:

        if len(chunk) <= max_size:
            final_chunks.append(chunk)
            continue

        for start in range(0, len(chunk), max_size):
            final_chunks.append(
                chunk[start:start + max_size].strip()
            )

    return [
        chunk
        for chunk in final_chunks
        if chunk
    ]


def create_hierarchy(elements: list) -> list:
    """
    Create parent sections and child chunks.
    """

    hierarchy = []

    current_parent = None
    parent_index = -1
    child_index = 0

    for element in elements:

        element_type = element["element_type"]

        # ---------------------------------------------------------
        # NEW SECTION
        # ---------------------------------------------------------

        if element_type == "heading":

            parent_index += 1

            current_parent = {
                "parent_id": f"parent_{parent_index:04d}",
                "section": element["content"],
                "page": element["page"],
                "children": [],
            }

            hierarchy.append(current_parent)

            child_index = 0

            continue

        # ---------------------------------------------------------
        # SAFETY: CONTENT BEFORE FIRST HEADING
        # ---------------------------------------------------------

        if current_parent is None:

            parent_index += 1

            current_parent = {
                "parent_id": f"parent_{parent_index:04d}",
                "section": "Introduction",
                "page": element["page"],
                "children": [],
            }

            hierarchy.append(current_parent)

        # ---------------------------------------------------------
        # SPECIAL ELEMENTS
        #
        # These are NEVER split.
        # ---------------------------------------------------------

        if element_type in {
            "table",
            "equation",
            "image",
        }:

            child = {
                "child_id": (
                    f"child_{parent_index:04d}_"
                    f"{child_index:04d}"
                ),
                "embedding_text": (
                    f"Section: {current_parent['section']}\n\n"
                    f"{element['content']}"
                ),
                "parent_id": current_parent["parent_id"],
                "element_type": element_type,
                "page": element["page"],
                "section": current_parent["section"],
                "content": element["content"],
            }

            if "caption" in element:
                child["caption"] = element["caption"]

            current_parent["children"].append(child)

            child_index += 1

            continue

        # ---------------------------------------------------------
        # NORMAL TEXT
        # ---------------------------------------------------------

        if element_type == "text":

            text_chunks = split_text(
                element["content"]
            )

            for text_chunk in text_chunks:

                child = {
                    "child_id": (
                        f"child_{parent_index:04d}_"
                        f"{child_index:04d}"
                    ),
                    "embedding_text": (
                        f"Section: {current_parent['section']}\n\n"
                        f"{text_chunk}"
                    ),
                    "parent_id": current_parent["parent_id"],
                    "element_type": "text",
                    "page": element["page"],
                    "section": current_parent["section"],
                    "content": text_chunk,
                }

                current_parent["children"].append(child)

                child_index += 1

    return hierarchy


def flatten_children(hierarchy: list) -> list:
    """
    Create a flat list of searchable child chunks.

    The hierarchy is still preserved through parent_id.
    """

    children = []

    for parent in hierarchy:

        for child in parent["children"]:

            children.append(child)

    return children


def main():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    print(f"[load] Reading: {INPUT_FILE}")

    elements = json.loads(
        INPUT_FILE.read_text(
            encoding="utf-8"
        )
    )

    print(
        f"[load] Elements: {len(elements)}"
    )

    hierarchy = create_hierarchy(elements)

    children = flatten_children(hierarchy)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "document": "MACHINE LEARNING.pdf",
        "parent_count": len(hierarchy),
        "child_count": len(children),
        "parents": hierarchy,
        "searchable_children": children,
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # -------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------

    element_counts = {}

    for child in children:

        element_type = child["element_type"]

        element_counts[element_type] = (
            element_counts.get(element_type, 0) + 1
        )

    print("\n[hierarchy] Created successfully")

    print(
        f"  Parent sections : {len(hierarchy)}"
    )

    print(
        f"  Child chunks    : {len(children)}"
    )

    print("\n[child breakdown]")

    for element_type, count in element_counts.items():

        print(
            f"  {element_type:10} : {count}"
        )

    print(
        f"\n[save] {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()