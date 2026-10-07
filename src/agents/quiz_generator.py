"""
src/agents/quiz_generator.py

The Quiz Generator agent.

Responsibilities:
  1. Generate quiz questions based on the explained topic
  2. Present questions to the user interactively via input()
  3. Grade each answer using the LLM as judge
  4. Return a QuizResult with score and identified weak areas

The same generate_questions and grade_answer functions are also reused
by the A2A service wrapper in src/a2a_services/quiz_service.py. The core
logic is identical in both modes; only the input/output mechanism changes
(terminal vs HTTP).

Architecture pattern:
  Two separate LLM calls with different purposes:
    - Generation call: creative, higher temperature, produces questions
    - Grading call: analytical, very low temperature, produces scores
  Separating these prevents the grader from being influenced by
  the generator's style or vice versa.
"""

import json
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import QuizQuestion, QuizResult, get_current_topic
from llm import get_chat_model, parse_json_content


# ─────────────────────────────────────────────────────────────────────────────
# Question generation
# ─────────────────────────────────────────────────────────────────────────────

GENERATION_PROMPT = """You are a quiz designer for a student learning programming.

Given a topic and explanation, generate {n} quiz questions that test
genuine understanding, not just the ability to repeat memorized phrases.

Good questions require the student to:
  - Apply a concept to a new situation
  - Explain WHY something works, not just WHAT it does
  - Identify edge cases or common mistakes
  - Compare related concepts

Return ONLY valid JSON with no prose or markdown:
{{
  "questions": [
    {{
      "question": "Clear, specific question text ending with ?",
      "expected_answer": "Model answer in 1-3 sentences",
      "difficulty": "easy|medium|hard"
    }}
  ]
}}

Rules:
  - Include at least one question about a common mistake or gotcha
  - expected_answer should be concise but complete
  - Avoid yes/no questions, ask for explanation or demonstration
  - Phrase every question as a direct question ("Why does...?", "What
    happens...?"), never as an instruction ("Explain...", "Describe...")
  - If a question includes a code snippet, put the code first; the question
    text must always END with a "?" character
"""

GRADING_PROMPT = """You are a fair teacher grading a student's answer.

Question: {question}
Model answer: {expected_answer}
Student's answer: {student_answer}

Grade the student's answer honestly. Be generous with partial credit:
  - Fundamentally correct with minor gaps: 0.7-0.9
  - Correct concept but imprecise: 0.5-0.7
  - Partially correct: 0.3-0.5
  - Fundamentally wrong: 0.0-0.2

Return ONLY valid JSON with no prose or markdown:
{{
  "correct": true,
  "score": 0.85,
  "feedback": "One specific sentence of feedback",
  "missing_concept": "Short noun phrase (2-6 words) naming the key concept missed, or empty string if answer is correct"
}}
"""


def generate_questions(topic: str, explanation: str, n: int = 3) -> list[dict]:
    """
    Call the LLM to generate n quiz questions about a topic.

    Args:
        topic:       The topic title being quizzed.
        explanation: The explanation the Explainer produced (context).
        n:           Number of questions to generate.

    Returns:
        List of question dicts with keys: question, expected_answer, difficulty.
        Falls back to one generic question if LLM output can't be parsed.
    """
    llm = get_chat_model(
        "quiz",
        temperature=0.4,   # Some creativity for varied questions
        json_mode=True,
    )

    prompt = GENERATION_PROMPT.format(n=n)
    try:
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"Topic: {topic}\n\nExplanation:\n{explanation}"),
        ])
    except Exception as e:
        print(f"[Quiz Generator] LLM call failed during question generation: {e}")
        # Return minimal fallback so the quiz can still run
        return [{
            "question": f"What is the main concept covered in {topic}?",
            "expected_answer": "See your study notes for this topic.",
            "difficulty": "medium",
        }]
    
    try:
        data = parse_json_content(response.content)
        questions = data.get("questions", [])
        if questions and isinstance(questions, list):
            return questions
    except (json.JSONDecodeError, KeyError):
        pass

    # Fallback: one generic question if parsing fails
    print("[Quiz Generator] Warning: could not parse questions, using fallback")
    return [{
        "question": f"In your own words, explain the key concept of {topic} and why it matters.",
        "expected_answer": "A clear explanation demonstrating conceptual understanding.",
        "difficulty": "medium",
    }]


def grade_answer(question: str, expected: str, student_answer: str) -> dict:
    """
    Use the LLM to grade a student's answer against the expected answer.

    Args:
        question:       The question that was asked.
        expected:       The model answer.
        student_answer: What the student wrote.

    Returns:
        Dict with keys: correct (bool), score (float), feedback (str),
        missing_concept (str).
        Returns a safe default if LLM output can't be parsed.
    """
    # Very low temperature, grading should be consistent and analytical
    # Separate "grader" role: a different model family from the question
    # writer, so grading isn't biased toward the generator's own phrasing.
    llm = get_chat_model("grader", temperature=0.1, json_mode=True)

    prompt = GRADING_PROMPT.format(
        question=question,
        expected_answer=expected,
        student_answer=student_answer,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
    except Exception as e:
        print(f"[Quiz Generator] LLM call failed during grading: {e}")
        # Return partial credit so the session can continue
        return {
            "correct": False,
            "score": 0.5,
            "feedback": "Could not grade answer due to a connection error.",
            "missing_concept": "",
        }
    
    try:
        return parse_json_content(response.content)
    except json.JSONDecodeError:
        # Safe default if grading fails
        return {
            "correct": False,
            "score": 0.0,
            "feedback": "Could not grade automatically, please review manually.",
            "missing_concept": "",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Interactive quiz runner
# ─────────────────────────────────────────────────────────────────────────────

def run_quiz(topic: str, explanation: str) -> QuizResult:
    """
    Run a complete interactive quiz on a topic.

    Generates questions, collects answers via input(), grades each,
    and returns a QuizResult.

    The same generate_questions and grade_answer functions back the A2A
    service wrapper in src/a2a_services/quiz_service.py. The quiz logic
    is identical; only how answers are collected changes.

    Args:
        topic:       The topic being quizzed.
        explanation: The Explainer's output (context for question generation).

    Returns:
        QuizResult with questions, scores, and identified weak areas.
    """
    print(f"\n{'='*60}")
    print(f"Quiz: {topic}")
    print(f"{'='*60}")
    print("Answer each question in your own words. Press Enter to submit.\n")

    questions_data = generate_questions(topic, explanation, n=3)
    graded_questions = []
    total_score = 0.0
    weak_areas = []

    for i, q_data in enumerate(questions_data, 1):
        question_text = q_data["question"]
        expected = q_data["expected_answer"]
        difficulty = q_data.get("difficulty", "medium")

        print(f"Question {i} [{difficulty}]: {question_text}")
        user_answer = input("Your answer: ").strip()

        # Handle empty answers
        if not user_answer:
            user_answer = "(no answer provided)"

        print("Grading...")
        grade = grade_answer(question_text, expected, user_answer)

        score = float(grade.get("score", 0.0))
        correct = bool(grade.get("correct", False))
        feedback = grade.get("feedback", "")
        missing = grade.get("missing_concept", "")

        total_score += score

        # Show result
        status = "✓" if correct else "✗"
        print(f"{status} Score: {score:.0%}, {feedback}\n")

        if missing:
            weak_areas.append(missing)

        graded_questions.append(QuizQuestion(
            question=question_text,
            expected_answer=expected,
            user_answer=user_answer,
            correct=correct,
            feedback=feedback,
            score=score,
        ))

    # Calculate overall score
    avg_score = total_score / len(questions_data) if questions_data else 0.0
    correct_count = sum(1 for q in graded_questions if q.correct)

    print(f"{'='*60}")
    print(f"Quiz complete! Score: {avg_score:.0%} "
          f"({correct_count}/{len(graded_questions)} correct)")
    if weak_areas:
        print(f"Areas to review: {', '.join(set(weak_areas))}")
    print(f"{'='*60}\n")

    return QuizResult(
        topic=topic,
        questions=graded_questions,
        score=avg_score,
        weak_areas=list(set(weak_areas)),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# The LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

def quiz_generator_node(state: dict) -> dict:
    """
    LangGraph node: Quiz Generator

    Reads:
        state["roadmap"]             : to get the current topic
        state["current_topic_index"]: which topic we're on
        state["messages"]            : to extract the explanation

    Writes:
        state["quiz_results"]        : appends the new QuizResult
        state["weak_areas"]          : accumulated weak areas (deduplicated)
        state["error"]               : error string on failure
    """
    topic = get_current_topic(state)
    if topic is None:
        return {"error": "No current topic, Curriculum Planner must run first"}

    # Extract the most recent explanation from messages
    # The Explainer's final response is the last AIMessage with no tool calls
    from langchain_core.messages import AIMessage
    messages = state.get("messages", [])
    explanation = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            explanation = msg.content
            break

    if not explanation:
        print("[Quiz Generator] Warning: no explanation found, generating generic quiz")
        explanation = f"Topic: {topic.title}. {topic.description}"

    print(f"\n[Quiz Generator] Generating quiz for: '{topic.title}'")
    quiz_result = run_quiz(topic.title, explanation)

    # Accumulate results
    existing_results = state.get("quiz_results", [])
    all_weak_areas = list(set(
        state.get("weak_areas", []) + quiz_result.weak_areas
    ))

    return {
        "quiz_results": existing_results + [quiz_result],
        "weak_areas": all_weak_areas,
        "error": None,
        # Pass core state through explicitly, LangGraph 1.1.0 state propagation workaround
        "roadmap": state.get("roadmap"),
        "current_topic_index": state.get("current_topic_index", 0),
        "session_id": state.get("session_id", ""),
    }
