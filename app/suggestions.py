def follow_up_questions(result: dict) -> list[str]:
    """Create useful, evidence-aware follow-ups without adding another model request."""
    route = result.get("route", "code_search")
    documents = result.get("documents", [])
    paths: list[str] = []
    symbols: list[str] = []
    for document in documents:
        path = str(document.metadata.get("path", ""))
        symbol = str(document.metadata.get("symbol", ""))
        if path and path not in paths:
            paths.append(path)
        if symbol and symbol != "module" and symbol not in symbols:
            symbols.append(symbol)
    focus = f"`{symbols[0]}`" if symbols else (f"`{paths[0]}`" if paths else "this component")
    related = f"`{paths[1]}`" if len(paths) > 1 else "the surrounding components"
    templates = {
        "architecture": [
            f"How does {focus} interact with {related}?",
            "Can you trace this data flow from entry point to final response?",
            "Which configuration settings control this architecture?",
        ],
        "debugging": [
            f"What failure conditions should I check in {focus}?",
            "Which logs and tests would help diagnose this issue?",
            "What is the safest code change to address this behavior?",
        ],
        "code_search": [
            f"Where is {focus} called from?",
            f"What tests cover {focus}?",
            "What related implementation should I inspect next?",
        ],
    }
    return templates.get(route, templates["code_search"])
