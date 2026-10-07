"""
src/observability/langfuse_setup.py

Langfuse observability setup for the Learning Accelerator.

Provides a single function, get_langfuse_config(), that returns
the LangGraph run config with a Langfuse callback handler attached.

Usage in main.py:
    from observability.langfuse_setup import get_langfuse_config
    config = get_langfuse_config(session_id)
    graph.invoke(state, config=config)

That single change captures:
  - Every agent node execution (start time, end time, status)
  - Every LLM call (model, prompt, response, tokens, latency)
  - Every tool call (name, args, result, latency)
  - Session metadata (session_id, user_id, tags)

No changes needed in agent code.
Everything is observed via the callback system automatically.
"""

import os

from llm import ROLE_MODELS, model_for


def _langfuse_configured() -> bool:
    """
    Check if Langfuse credentials are available in the environment.

    Returns False if keys are missing or empty, in that case,
    the system runs without observability rather than crashing.
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    return bool(public_key and secret_key)


def get_langfuse_handler(session_id: str, user_id: str = "local"):
    """
    Create a Langfuse callback handler for a session.

    Args:
        session_id: The study session ID (used as Langfuse session_id).
                    Groups all traces from one study session together.
        user_id:    Optional user identifier for filtering in the UI.

    Returns:
        A configured CallbackHandler, or None if Langfuse is not set up.
        Callers should handle None gracefully.
    """
    if not _langfuse_configured():
        return None

    try:
        from langfuse.langchain import CallbackHandler

        handler = CallbackHandler(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
            session_id=session_id,
            user_id=user_id,
            # Tags appear in the Langfuse UI, useful for filtering
            # traces from this tutorial vs other projects
            tags=["learning-accelerator", "openrouter"],
            metadata={
                # Each agent role uses its own model; Langfuse records the
                # exact model per LLM call, this is just the session overview.
                "models": {role: model_for(role) for role in ROLE_MODELS},
                "framework": "langgraph",
            },
        )
        return handler
    except ImportError:
        print("[Observability] langfuse not installed. Run: pip install langfuse")
        return None
    except Exception as e:
        print(f"[Observability] Failed to create Langfuse handler: {e}")
        return None


def get_langfuse_config(
    session_id: str,
    user_id: str = "local",
    extra_config: dict | None = None,
) -> dict:
    """
    Build a complete LangGraph run config with Langfuse observability.

    This is the main function to use in main.py. It merges:
      - thread_id for checkpointing
      - Langfuse callback handler (if configured)
      - Any additional config you pass in

    Args:
        session_id:   The study session ID.
        user_id:      Optional user identifier.
        extra_config: Any additional LangGraph config to merge in.

    Returns:
        A dict ready to pass as `config` to graph.invoke().

    Example:
        config = get_langfuse_config(session_id)
        result = graph.invoke(state, config=config)
        # All agent calls now appear in Langfuse UI
    """
    config = {
        "configurable": {"thread_id": session_id},
    }

    # Merge any extra config
    if extra_config:
        config.update(extra_config)

    # Attach Langfuse handler if available
    handler = get_langfuse_handler(session_id, user_id)
    if handler:
        config["callbacks"] = [handler]
        print(f"[Observability] Tracing session {session_id} → "
              f"{os.getenv('LANGFUSE_HOST', 'http://localhost:3000')}")
    else:
        print("[Observability] Langfuse not configured. Running without tracing.")

    return config


def flush_langfuse() -> None:
    """
    Flush any pending Langfuse events before process exit.

    Langfuse sends traces asynchronously in a background thread.
    Call this at the end of main.py to ensure all traces are sent
    before the process exits.

    If Langfuse is not configured, this is a no-op.
    """
    if not _langfuse_configured():
        return

    try:
        from langfuse import Langfuse
        Langfuse().flush()
    except Exception:
        pass  # Best-effort flush, don't crash on exit
