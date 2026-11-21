"""
LLM access for the proposed method's components.

Every component uses the same model and reasoning effort as the published
planner, so P differs from the baselines in structure, not in model.
"""

import json
from utils.llm_models import llm_effort, llm_model

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

MODEL = llm_model("spatial_reasoning.components")
REASONING_EFFORT = llm_effort("spatial_reasoning.components")
RETRIES = 2


class InvalidResponse(ValueError):
    """The LLM's answer still failed validation after every retry."""


def make_llm():
    from utils.agm_helper import GIBDChat
    return GIBDChat(model_name=MODEL, reasoning_effort=REASONING_EFFORT)


def parse_json(text):
    """The first JSON object in *text*, tolerating code fences and preamble."""
    cleaned = (text or "").replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    if start < 0:
        raise InvalidResponse("no JSON object in the response")
    obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    return obj


def ask_json(system, user, validate, *, answer=None, retries=RETRIES):
    """Ask for a JSON object; return validate(obj). *validate* raises
    InvalidResponse with a readable message, which is sent back so the model
    can correct itself. *answer* replaces the plain LLM call (used for the
    tool-using hypotheses agent and in tests): answer(messages) -> str."""
    if answer is None:
        llm = make_llm()
        answer = lambda msgs: llm.invoke(msgs).content
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    problem = None
    for _ in range(retries + 1):
        text = answer(messages)
        try:
            return validate(parse_json(text))
        except (InvalidResponse, ValueError) as e:
            problem = str(e)
            messages += [AIMessage(content=text), HumanMessage(content=(
                f"Your previous answer was not valid: {problem}\n"
                "Return the corrected JSON object only."))]
    raise InvalidResponse(problem)
