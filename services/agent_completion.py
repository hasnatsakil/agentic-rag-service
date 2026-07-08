"""
LLM completion wrapper for the OpenRouter API.

This module exposes :func:`agent_complete`, a single unified function for
making chat completion calls through OpenRouter.  It handles:

- **Model selection** — chooses the appropriate model based on task type
  (standard, grading, or hallucination-checking).
- **Reasoning configuration** — injects OpenRouter's ``reasoning`` parameter
  with provider-specific handling for Anthropic models.
- **Structured logging** — records model, provider, runtime, and a preview
  of the conversation using :mod:`structlog`.

Default models are configured via :class:`~config.openrouter_settings.RouterModel`
and :class:`~config.openrouter_settings.RouterGradingModel` enumerations.
"""

import time
from typing import List, Optional

import structlog
from openai import ChatCompletion

from config.openrouter_settings import RouterModel, RouterGradingModel, RouterConfig, client

logger = structlog.get_logger(__name__)

# ------------------------------------------------------------------ #
#  Default model selection                                            #
# ------------------------------------------------------------------ #

#: Default model used for standard answer generation.
MODEL = RouterModel.MINIMAX25.value

#: Default model used for relevance grading (lightweight binary classifier).
GRADE_MODEL = RouterGradingModel.LLAMA.value

#: Default model used for hallucination detection.
HALLUCINATION_MODEL = RouterModel.QWEN3NEXT.value


def agent_complete(
    messages: List[dict],
    tools: Optional[List[dict]] = None,
    model: Optional[str] = None,
    is_grading: bool = False,
    is_hallucination: bool = False,
    max_tokens: int = 500,
    temperature: float = 0.3,
    reasoning_effort: str = "medium",
    usage_instance=None,
) -> ChatCompletion:
    """Call the OpenRouter chat completions API and return the raw completion.

    Selects the appropriate model based on the task type flags, builds the
    router configuration (provider order, fallback models, optional reasoning),
    and logs structured metadata about the request and response.

    Args:
        messages: A list of OpenAI-format message dicts
            (``{"role": ..., "content": ...}``).
        tools: Optional list of OpenAI-compatible tool definitions to pass
            to the model, enabling function-calling.  ``None`` disables
            tool use.
        model: Explicit model ID to use.  When ``None``, the model is chosen
            automatically based on ``is_grading`` and ``is_hallucination``.
        is_grading: When ``True``, uses :data:`GRADE_MODEL` (a lightweight
            binary classifier for relevance grading).  Ignored if ``model``
            is explicitly provided.
        is_hallucination: When ``True``, uses :data:`HALLUCINATION_MODEL`
            for grounded-answer verification.  Takes precedence over
            ``is_grading`` when both are ``True``.  Ignored if ``model``
            is explicitly provided.
        max_tokens: Maximum number of tokens to generate. Defaults to ``500``.
        temperature: Sampling temperature for the completion.  Lower values
            produce more deterministic output. Defaults to ``0.3``.
        reasoning_effort: OpenRouter reasoning effort level.  One of
            ``"low"``, ``"medium"``, ``"high"``, or ``"xhigh"``.
            Defaults to ``"medium"``.  For Anthropic models, this is
            converted to a ``max_tokens``-based budget instead.
        usage_instance: Reserved for future usage tracking integration.
            Currently unused.

    Returns:
        The raw :class:`openai.ChatCompletion` object returned by the API.
        The caller is responsible for extracting ``choices[0].message.content``
        or ``choices[0].message.tool_calls`` as appropriate.

    Example::

        completion = agent_complete(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is RAG?"},
            ]
        )
        answer = completion.choices[0].message.content
    """
    str_time = time.time()

    # Determine model from task-type flags if not explicitly provided.
    if model is None:
        if is_hallucination:
            model = HALLUCINATION_MODEL
        elif is_grading:
            model = GRADE_MODEL
        else:
            model = MODEL

    router_config = RouterConfig.config(
        model,
        is_grading=is_grading,
        is_hallucination=is_hallucination,
    )

    # Inject reasoning configuration; Anthropic uses token budget instead of effort string.
    # if reasoning_effort:
    #     router_config["reasoning"] = {"effort": reasoning_effort}
    #     if "anthropic" in model.lower():
    #         router_config["reasoning"] = {
    #             "max_tokens": max_tokens // 1.5
    #         }

    # Specifically for Gradding and Hallucination
    if reasoning_effort and not (is_grading or is_hallucination):
        router_config["reasoning"] = {"effort": reasoning_effort}
        if "anthropic" in model.lower():
            router_config["reasoning"] = {
                "max_tokens": max_tokens // 1.5
            }
    
    # if is_grading or is_hallucination:
    #     router_config["include_reasoning"] = False

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=router_config,
    )
    choice = completion.choices[0]
    runtime = time.time() - str_time

    # Truncate long system prompts in the log to avoid noise.
    sys_msg = messages[0]["content"] if messages else ""
    sys_summary = f"...{sys_msg[-1000:]}" if len(sys_msg) > 1000 else sys_msg

    logger.info(
        "[Agentic] Completion done",
        completion_message=choice.message,
        message_system_summary=sys_summary,
        messages_preview=messages[1:][-5:],  # Last 5 non-system messages.
        message_len=len(messages),
        max_tokens=max_tokens,
        model_completion=completion.model,
        model_requested=model,
        provider=completion.provider,
        runtime=runtime,
    )

    return completion
