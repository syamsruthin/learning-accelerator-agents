"""
tests/conftest.py

Shared pytest configuration, fixtures, and markers.

Defines two test markers:
  @pytest.mark.unit , fast, no external deps (default)
  @pytest.mark.eval , slow, makes real OpenRouter calls, LLM-as-judge

Run only unit tests (fast, development):
  pytest tests/ -m "not eval" -v

Run only eval tests (slow, before releases):
  pytest tests/test_eval.py -m eval -v -s

Run everything:
  pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

# Ensure src/ is on path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def pytest_configure(config):
    """Register custom markers so pytest doesn't warn about unknown marks."""
    config.addinivalue_line(
        "markers",
        "eval: marks tests as evaluation tests making real OpenRouter calls (deselect with -m 'not eval')"
    )
    config.addinivalue_line(
        "markers",
        "unit: marks tests as fast unit tests with no external dependencies"
    )


@pytest.fixture(autouse=True)
def openrouter_key(request, monkeypatch):
    """
    Unit tests build LLM clients (e.g. the CrewAI Study Buddy crew) but never
    call them, so a placeholder key is enough. Eval tests make real calls and
    are skipped when no real key is configured.
    """
    import os
    if request.node.get_closest_marker("eval"):
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env")
        if not os.getenv("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set, skipping eval test")
    elif not os.getenv("OPENROUTER_API_KEY"):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-placeholder-key")


@pytest.fixture
def sample_roadmap():
    """
    A minimal StudyRoadmap for use in tests.
    Avoids repeating roadmap construction across test files.
    """
    from graph.state import StudyRoadmap, Topic
    return StudyRoadmap(
        goal="Learn Python closures",
        total_weeks=2,
        topics=[
            Topic(
                title="Closures Explained",
                description="Understand how closures capture enclosing scope variables",
                estimated_minutes=60,
            ),
            Topic(
                title="Practical Closure Patterns",
                description="Apply closures to real problems: factories, memoisation",
                estimated_minutes=45,
                prerequisites=["Closures Explained"],
            ),
        ],
    )


@pytest.fixture
def sample_state(sample_roadmap):
    """
    A minimal AgentState dict for use in tests.
    Has a roadmap, session ID, and all required fields populated.
    """
    from graph.state import initial_state
    state = initial_state("Learn Python closures", "test-session-001")
    state["roadmap"] = sample_roadmap
    state["current_topic_index"] = 0
    return state


@pytest.fixture
def closures_note_content():
    """
    The content of the closures.md sample note.
    Used as retrieval context in faithfulness tests.
    """
    notes_path = Path(__file__).parent.parent / "study_materials/sample_notes/closures.md"
    if notes_path.exists():
        return notes_path.read_text(encoding="utf-8")
    # Fallback if file doesn't exist
    return """
# Python Closures

A closure is a nested function that remembers variables from its enclosing scope.

Three requirements:
1. A nested (inner) function
2. The inner function refers to a variable from the enclosing scope
3. The enclosing function returns the inner function

Example:
def make_counter(start=0):
    count = start
    def increment():
        nonlocal count
        count += 1
        return count
    return increment
"""
