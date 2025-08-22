"""Glossary management module for handling JSON-based glossary storage."""

import json
from pathlib import Path
from typing import Dict


class GlossaryManager:
    """Manages glossary operations with JSON file storage."""

    def __init__(self, glossary_path: str = None):
        """Initialize the glossary manager.

        Args:
            glossary_path: Path to the glossary JSON file. If None, uses default path.
        """
        if glossary_path is None:
            # Default to glossary.json in the same directory as this file
            self.glossary_path = Path(__file__).parent / "glossary.json"
        else:
            self.glossary_path = Path(glossary_path)

        self._glossary_cache = None

    def load_glossary(self) -> Dict[str, Dict[str, str]]:
        """Load glossary from JSON file.

        Returns:
            Dictionary containing the glossary sources.
        """
        if not self.glossary_path.exists():
            # Create empty glossary if file doesn't exist
            self._create_empty_glossary()
            return {}

        try:
            with open(self.glossary_path, encoding="utf-8") as f:
                self._glossary_cache = json.load(f)
                return self._glossary_cache
        except (OSError, json.JSONDecodeError) as e:
            print(f"Error loading glossary from {self.glossary_path}: {e}")
            return {}

    def save_glossary(self, glossary: Dict[str, Dict[str, str]]) -> bool:
        """Save glossary to JSON file.

        Args:
            glossary: Dictionary containing the glossary sources.

        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            # Ensure directory exists
            self.glossary_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.glossary_path, "w", encoding="utf-8") as f:
                json.dump(glossary, f, indent=2, ensure_ascii=False)

            # Update cache
            self._glossary_cache = glossary.copy()
            return True
        except OSError as e:
            print(f"Error saving glossary to {self.glossary_path}: {e}")
            return False

    def add_source(self, source: str, target: str, note: str = "") -> bool:
        """Add or update a source in the glossary.

        Args:
            source: The English source to translate.
            target: The translation target.
            note: Optional note about when to use this translation.

        Returns:
            True if added successfully, False otherwise.
        """
        glossary = self.load_glossary()

        glossary[source.lower()] = {"target": target, "note": note}

        return self.save_glossary(glossary)

    def remove_source(self, source: str) -> bool:
        """Remove a source from the glossary.

        Args:
            source: The source to remove.

        Returns:
            True if removed successfully, False if source not found or error occurred.
        """
        glossary = self.load_glossary()

        source_lower = source.lower()
        if source_lower not in glossary:
            return False

        del glossary[source_lower]
        return self.save_glossary(glossary)

    def get_source(self, source: str) -> Dict[str, str] | None:
        """Get a specific source from the glossary.

        Args:
            source: The source to look up.

        Returns:
            Dictionary with 'target' and 'note' keys, or None if not found.
        """
        glossary = self.load_glossary()
        return glossary.get(source.lower())

    def update_source(self, source: str, target: str = None, note: str = None) -> bool:
        """Update an existing source in the glossary.

        Args:
            source: The source to update.
            target: New translation target (optional).
            note: New note (optional).

        Returns:
            True if updated successfully, False if source not found or error occurred.
        """
        glossary = self.load_glossary()

        source_lower = source.lower()
        if source_lower not in glossary:
            return False

        if target is not None:
            glossary[source_lower]["target"] = target

        if note is not None:
            glossary[source_lower]["note"] = note

        return self.save_glossary(glossary)

    def get_all_sources(self) -> Dict[str, Dict[str, str]]:
        """Get all sources from the glossary.

        Returns:
            Complete glossary dictionary.
        """
        return self.load_glossary()

    def search_sources(
        self, search_text: str, search_in_notes: bool = True
    ) -> Dict[str, Dict[str, str]]:
        """Search for sources containing the search text.

        Args:
            search_text: Text to search for.
            search_in_notes: Whether to also search in notes.

        Returns:
            Dictionary of matching sources.
        """
        glossary = self.load_glossary()
        matches = {}
        search_lower = search_text.lower()

        for source, data in glossary.items():
            if (
                search_lower in source.lower()
                or search_lower in data["target"].lower()
                or (search_in_notes and search_lower in data["note"].lower())
            ):
                matches[source] = data

        return matches

    def _create_empty_glossary(self):
        """Create an empty glossary file."""
        self.save_glossary({})


# Convenience functions for backward compatibility
def load_glossary(glossary_path: str = None) -> Dict[str, Dict[str, str]]:
    """Load glossary from JSON file."""
    manager = GlossaryManager(glossary_path)
    return manager.load_glossary()


def save_glossary(
    glossary: Dict[str, Dict[str, str]], glossary_path: str = None
) -> bool:
    """Save glossary to JSON file."""
    manager = GlossaryManager(glossary_path)
    return manager.save_glossary(glossary)


def add_glossary_source(
    source: str, target: str, note: str = "", glossary_path: str = None
) -> bool:
    """Add a source to the glossary."""
    manager = GlossaryManager(glossary_path)
    return manager.add_source(source, target, note)
