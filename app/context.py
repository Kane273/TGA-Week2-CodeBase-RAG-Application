from langchain_core.documents import Document


def retrieval_policy(route: str, question: str) -> dict[str, int]:
    """Spend retrieval work in proportion to the question's likely complexity."""
    if route == "architecture":
        return {"hops": 3, "top_k": 10, "queries": 3}
    if route == "debugging":
        return {"hops": 2, "top_k": 8, "queries": 2}
    if len(question.split()) <= 8:
        return {"hops": 1, "top_k": 5, "queries": 1}
    return {"hops": 2, "top_k": 7, "queries": 2}


def build_bounded_context(documents: list[Document], max_characters: int) -> tuple[str, list[Document]]:
    """Deduplicate identical chunks and include only complete chunks within a hard budget."""
    blocks: list[str] = []
    selected: list[Document] = []
    seen_content: set[str] = set()
    used = 0
    for document in documents:
        normalized = " ".join(document.page_content.split())
        if not normalized or normalized in seen_content:
            continue
        metadata = document.metadata
        block = (f'[{metadata["path"]}:{metadata["start_line"]}-{metadata["end_line"]}]\n'
                 f"{document.page_content}")
        if used + len(block) > max_characters:
            continue
        seen_content.add(normalized)
        blocks.append(block)
        selected.append(document)
        used += len(block) + 2
    return "\n\n".join(blocks), selected
