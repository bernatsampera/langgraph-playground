def format_glossary(glossary: dict[str, dict[str, str]]) -> str:
    """Format the glossary to be used in the prompt."""
    return "\n".join(
        [
            f"{key}: {value['value']} ({value['comment']})"
            for key, value in glossary.items()
        ]
    )
