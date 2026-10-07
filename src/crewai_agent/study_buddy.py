"""
src/crewai_agent/study_buddy.py

A CrewAI-based study buddy agent exposed as an A2A service.

This demonstrates cross-framework agent interoperability:
  - The service is built with CrewAI (not LangGraph)
  - It's exposed via the same A2A protocol as the Quiz Service
  - The LangGraph Progress Coach calls it via A2A
  - Neither framework knows about the other's internals

The Study Buddy's role: when a student scores poorly, the Progress
Coach can request supplementary explanation and additional questions
from a "different perspective", the CrewAI Study Buddy.

Run standalone:
  python src/crewai_agent/study_buddy.py

Agent Card:
  http://localhost:9002/.well-known/agent-card.json
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add src/ to path so the shared model config in src/llm.py is importable
# when this file is run directly (python src/crewai_agent/study_buddy.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from a2a.utils import new_agent_text_message
from crewai import Agent, Crew, LLM, Process, Task
from crewai.events import crewai_event_bus
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
)
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from llm import OPENROUTER_BASE_URL, get_api_key, model_for
from token_usage import record_usage


# ─────────────────────────────────────────────────────────────────────────────
# Token usage logging
#
# CrewAI doesn't use LangChain callbacks, so the LLM factory's usage callback
# doesn't see these calls. Instead we listen on CrewAI's event bus, which
# emits one event per LLM call (a Study Buddy run is usually 2 calls: one
# deciding to use the topic_analyser tool, one writing the answer).
# ─────────────────────────────────────────────────────────────────────────────

_llm_call_started: dict[str, float] = {}   # call_id -> start time


@crewai_event_bus.on(LLMCallStartedEvent)
def _on_llm_call_started(source, event: LLMCallStartedEvent) -> None:
    _llm_call_started[event.call_id] = time.perf_counter()


@crewai_event_bus.on(LLMCallCompletedEvent)
def _on_llm_call_completed(source, event: LLMCallCompletedEvent) -> None:
    start = _llm_call_started.pop(event.call_id, None)
    usage = event.usage or {}
    record_usage(
        "study_buddy",
        event.model or model_for("study_buddy"),
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        latency_s=time.perf_counter() - start if start is not None else None,
    )


@crewai_event_bus.on(LLMCallFailedEvent)
def _on_llm_call_failed(source, event: LLMCallFailedEvent) -> None:
    start = _llm_call_started.pop(event.call_id, None)
    record_usage(
        "study_buddy",
        event.model or model_for("study_buddy"),
        0,
        0,
        latency_s=time.perf_counter() - start if start is not None else None,
        status="error",
        error=str(event.error)[:300],
    )


# ─────────────────────────────────────────────────────────────────────────────
# CrewAI Tools
#
# Tools give the CrewAI agent capabilities beyond just LLM calls.
# Each tool is a class with a _run() method.
# BaseTool handles the schema generation and validation.
# ─────────────────────────────────────────────────────────────────────────────

class TopicAnalyserInput(BaseModel):
    """Input schema for the topic analyser tool."""
    topic: str = Field(description="The topic to analyse")
    weak_areas: list[str] = Field(
        default_factory=list,
        description="Weak areas the student struggled with",
    )


class TopicAnalyserTool(BaseTool):
    """
    Analyses a topic and weak areas to produce a structured study plan.

    CrewAI tool, called by the Study Buddy agent to structure its response.
    In production this might call an API or database. For the tutorial,
    it produces structured analysis from the inputs.
    """
    name: str = "topic_analyser"
    description: str = (
        "Analyse a study topic and the student's weak areas to produce "
        "a structured list of key concepts to focus on."
    )
    args_schema: type[BaseModel] = TopicAnalyserInput

    def _run(self, topic: str, weak_areas: list[str] | None = None) -> str:
        """
        Return a structured analysis of what to focus on.

        In a production system this might query a knowledge graph,
        call an external API, or read from a curriculum database.
        For the tutorial, it returns structured guidance.
        """
        areas = weak_areas or []
        focus_items = areas if areas else [f"Core concepts of {topic}"]

        analysis = {
            "topic": topic,
            "focus_areas": focus_items,
            "suggested_approach": (
                f"Start with the fundamentals of {topic}, then address: "
                f"{', '.join(focus_items)}."
            ),
            "study_tip": (
                "Try explaining the concept out loud in your own words. "
                "If you can teach it simply, you understand it."
            ),
        }
        return json.dumps(analysis)


# ─────────────────────────────────────────────────────────────────────────────
# CrewAI Agent and Crew builder
#
# This function creates a fresh Crew for each incoming A2A task.
# Creating per-request avoids state leakage between tasks.
# ─────────────────────────────────────────────────────────────────────────────

def build_study_buddy_crew(
    topic: str,
    explanation: str,
    weak_areas: list[str],
) -> Crew:
    """
    Build a CrewAI crew for a specific study assistance request.

    Returns a Crew ready to kickoff(). Called once per A2A task.

    Args:
        topic:       The topic the student is studying.
        explanation: The Explainer's output (context for the Study Buddy).
        weak_areas:  Concepts the student struggled with in the quiz.
    """
    topic_analyser = TopicAnalyserTool()

    # Configure CrewAI to use OpenRouter. The "openrouter/" prefix routes
    # the call through LiteLLM's OpenRouter provider. Only the model config
    # is shared with the LangGraph side (src/llm.py), not any LangChain code.
    llm = LLM(
        model=f"openrouter/{model_for('study_buddy')}",
        base_url=OPENROUTER_BASE_URL,
        api_key=get_api_key(),
        temperature=0.5,   # A "fresh angle" benefits from some variety
    )

    study_buddy_agent = Agent(
        role="Study Buddy",
        goal=(
            "Provide clear, encouraging supplementary explanations that help "
            "students understand difficult concepts from a fresh angle."
        ),
        backstory=(
            "You are an experienced tutor who has helped hundreds of students "
            "master programming concepts. You specialise in finding alternative "
            "explanations and analogies that make difficult ideas click."
        ),
        llm=llm,
        tools=[topic_analyser],
        verbose=False,
        allow_delegation=False,
    )

    weak_areas_text = (
        f"The student struggled with: {', '.join(weak_areas)}"
        if weak_areas
        else "No specific weak areas identified."
    )

    study_task = Task(
        description=(
            f"A student is studying '{topic}'. Here is the explanation they received:\n\n"
            f"{explanation[:1000]}\n\n"  # Limit to 1000 chars to stay within context
            f"{weak_areas_text}\n\n"
            "First use the topic_analyser tool to structure your approach. "
            "Then provide: "
            "1) A fresh analogy that explains the core concept differently, "
            "2) One concrete example that illustrates the weak area(s), "
            "3) One practical tip for remembering this concept. "
            "Keep your response concise and encouraging (150-250 words)."
        ),
        agent=study_buddy_agent,
        expected_output=(
            "A structured study assistance response with a fresh analogy, "
            "a concrete example targeting weak areas, and a memory tip."
        ),
    )

    return Crew(
        agents=[study_buddy_agent],
        tasks=[study_task],
        process=Process.sequential,
        verbose=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent Card
# ─────────────────────────────────────────────────────────────────────────────

STUDY_BUDDY_SKILL = AgentSkill(
    id="supplementary_study_assistance",
    name="Supplementary Study Assistance",
    description=(
        "Provides supplementary study assistance when a student needs "
        "a different explanation angle. Given a topic, the original explanation, "
        "and any weak areas, returns a fresh analogy, a targeted example, "
        "and a memory tip. Built with CrewAI."
    ),
    tags=["study", "tutoring", "explanation", "crewai"],
    examples=[
        "Help a student understand Python closures from a different angle",
        "Provide supplementary explanation for decorator weak areas",
    ],
)

STUDY_BUDDY_CARD = AgentCard(
    name="CrewAI Study Buddy",
    description=(
        "A supplementary learning assistant built with CrewAI. "
        "Provides alternative explanations and targeted examples when "
        "the primary explanation didn't land. "
        "Framework-agnostic: connects via A2A protocol."
    ),
    url="http://localhost:9002/",
    version="1.0.0",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(streaming=False),
    skills=[STUDY_BUDDY_SKILL],
)


# ─────────────────────────────────────────────────────────────────────────────
# A2A Agent Executor wrapping CrewAI
# ─────────────────────────────────────────────────────────────────────────────

class StudyBuddyExecutor(AgentExecutor):
    """
    Bridges the A2A protocol to CrewAI execution.

    This is the key integration point: the A2A framework calls execute(),
    we parse the request, run a CrewAI crew, and emit the result.

    The LangGraph system has no idea this is CrewAI.
    The CrewAI crew has no idea it's serving an A2A request.
    """

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Handle an incoming A2A study assistance task."""

        # ── Parse request ─────────────────────────────────────────────
        # get_user_input() joins the text parts of the incoming message
        # (a2a-sdk 0.3 wraps each part as Part(root=TextPart(...))).
        request_text = context.get_user_input()

        try:
            request_data = json.loads(request_text)
        except json.JSONDecodeError:
            request_data = {"topic": request_text}

        topic = request_data.get("topic", "General Topic")
        explanation = request_data.get("explanation", "")
        weak_areas = request_data.get("weak_areas", [])

        print(f"[Study Buddy A2A] Request: topic='{topic}', "
              f"weak_areas={weak_areas}")

        # ── Run CrewAI in thread pool ─────────────────────────────────
        # CrewAI's kickoff() is synchronous. We run it in a thread pool
        # so it doesn't block the async event loop.
        try:
            crew = build_study_buddy_crew(topic, explanation, weak_areas)
            crew_result = await asyncio.to_thread(crew.kickoff)

            # Extract the string output from CrewAI result
            result_text = str(crew_result)
            if hasattr(crew_result, "raw"):
                result_text = crew_result.raw

            result = {
                "source":      "crewai_study_buddy",
                "topic":       topic,
                "weak_areas":  weak_areas,
                "assistance":  result_text,
                "status":      "complete",
            }
            print(f"[Study Buddy A2A] Task complete ({len(result_text)} chars)")

        except Exception as e:
            print(f"[Study Buddy A2A] CrewAI error: {e}")
            result = {
                "source":     "crewai_study_buddy",
                "topic":      topic,
                "assistance": (
                    "I encountered an issue generating supplementary help "
                    f"for '{topic}'. Please review the original explanation "
                    "and try again."
                ),
                "status": "error",
                "error":  str(e),
            }

        # ── Emit result ───────────────────────────────────────────────
        # new_agent_text_message() sets the required message_id and ties
        # the reply to this request's task/context IDs.
        await event_queue.enqueue_event(
            new_agent_text_message(
                json.dumps(result, indent=2),
                context_id=context.context_id,
                task_id=context.task_id,
            )
        )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Server setup
# ─────────────────────────────────────────────────────────────────────────────

def create_study_buddy_server():
    """Build the CrewAI Study Buddy A2A application."""
    request_handler = DefaultRequestHandler(
        agent_executor=StudyBuddyExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(
        agent_card=STUDY_BUDDY_CARD,
        http_handler=request_handler,
    )
    return app.build()


if __name__ == "__main__":
    print("[CrewAI Study Buddy] Starting on http://localhost:9002")
    print("[CrewAI Study Buddy] Agent Card: "
          "http://localhost:9002/.well-known/agent-card.json")
    print("[CrewAI Study Buddy] This is a CrewAI agent served via A2A")
    print("[CrewAI Study Buddy] Press Ctrl+C to stop\n")
    uvicorn.run(
        create_study_buddy_server(),
        host="0.0.0.0",
        port=9002,
        log_level="warning",
    )
