"""
src/llm.py

Central LLM factory for the Learning Accelerator.

Every LLM call in the system goes through OpenRouter, an OpenAI-compatible
gateway that routes to many model providers behind one API key.

Why one factory instead of a ChatOpenAI(...) in every agent?
  - Each agent role gets a model chosen for its job (see ROLE_MODELS)
  - Switching a model is a one-line .env change, no code edits
  - The OpenRouter base URL, API key and attribution headers live in
    exactly one place

Model choices per role are documented in claude-reasoning/.
Any role can be overridden with an env var, e.g.:
  OPENROUTER_MODEL_EXPLAINER=anthropic/claude-haiku-5.5
"""

import json
import os
import re


OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


# ─────────────────────────────────────────────────────────────────────────────
# Model per agent role
#
# Key = role name, value = (env var override, default OpenRouter model id).
# ─────────────────────────────────────────────────────────────────────────────

ROLE_MODELS: dict[str, tuple[str, str]] = {
    # One call per session, shapes the whole study plan: worth a strong model.
    "planner":     ("OPENROUTER_MODEL_PLANNER",     "anthropic/claude-sonnet-5.5"),
    # Multi-step tool-calling loop over the student's notes, user-facing teaching.
    "explainer":   ("OPENROUTER_MODEL_EXPLAINER",   "anthropic/claude-sonnet-5.5"),
    # Short structured JSON (3 questions), called once per topic.
    "quiz":        ("OPENROUTER_MODEL_QUIZ",        "anthropic/claude-haiku-5.5"),
    # LLM-as-judge on every answer: different model family from the question
    # writer, so the grader doesn't share the generator's blind spots.
    "grader":      ("OPENROUTER_MODEL_GRADER",      "openai/gpt-6-luna"),
    # 2-3 sentences of encouragement: cheapest capable model is enough.
    "coach":       ("OPENROUTER_MODEL_COACH",       "anthropic/claude-haiku-5.5"),
    # CrewAI Study Buddy: its job is a *fresh angle*, so a different family
    # from the Explainer gives genuinely different analogies.
    "study_buddy": ("OPENROUTER_MODEL_STUDY_BUDDY", "google/gemini-3.8-flash"),
    # DeepEval judge in tests/test_eval.py: strongest model, and independent
    # of the models being evaluated (Explainer is Anthropic).
    "eval_judge":  ("OPENROUTER_MODEL_EVAL_JUDGE",  "openai/gpt-6-sol"),
}


def model_for(role: str) -> str:
    """Resolve the OpenRouter model id for an agent role (env override first)."""
    env_var, default = ROLE_MODELS[role]
    return os.getenv(env_var) or default


def get_api_key() -> str:
    """Read the OpenRouter API key, failing loudly if it is missing."""
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file "
            "(get one at https://openrouter.ai/keys)."
        )
    return key


def get_chat_model(
    role: str,
    temperature: float = 0.3,
    json_mode: bool = False,
):
    """
    Build a LangChain chat model for an agent role, served via OpenRouter.

    Args:
        role:        Key in ROLE_MODELS, e.g. "planner", "explainer".
        temperature: Sampling temperature (low for structured output/grading).
        json_mode:   Request a JSON object response. Replaces Ollama's
                     format="json". Pair with parse_json_content() since some
                     providers still wrap JSON in markdown fences.
    """
    # Imported here, not at module top, so non-LangChain code (the CrewAI
    # Study Buddy) can use model_for()/get_api_key() without pulling in
    # langchain_openai. Returns a langchain_openai.ChatOpenAI.
    from langchain_openai import ChatOpenAI

    from token_usage_callback import TokenUsageCallbackHandler

    model = model_for(role)
    model_kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    return ChatOpenAI(
        model=model,
        base_url=OPENROUTER_BASE_URL,
        api_key=get_api_key(),
        temperature=temperature,
        timeout=120,
        max_retries=2,
        model_kwargs=model_kwargs,
        # Optional OpenRouter attribution headers (shown in your dashboard)
        default_headers={
            "HTTP-Referer": "https://github.com/learning-accelerator-agents",
            "X-Title": "Learning Accelerator",
        },
        # Logs input/output tokens for every call made with this model,
        # including each iteration of the Explainer's tool loop.
        # Runs alongside any callbacks passed via config (e.g. Langfuse).
        callbacks=[TokenUsageCallbackHandler(role, model)],
    )


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_json_content(content: str):
    """
    json.loads() that tolerates ```json fences around the payload.

    Raises json.JSONDecodeError like json.loads() so existing fallbacks
    in the agents keep working unchanged.
    """
    text = content.strip()
    match = _FENCE_RE.match(text)
    if match:
        text = match.group(1)
    return json.loads(text)
