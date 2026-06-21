from typing import List
from config.openrouter_settings import RouterModel, RouterConfig, client
from openai import ChatCompletion
import time
import structlog

logger = structlog.get_logger(__name__)
MODEL = RouterModel.MINIMAX25.value

def agent_complete(
        messages: List[dict],
        tools: List[dict] = None,
        model: str = MODEL,
        max_tokens: int = 500,
        temperature: float = 0.3,
        reasoning_effort: str = "medium",  # "low", "medium", "high", "xhigh"
        usage_instance=None,

) -> ChatCompletion:
    str_time = time.time()
    router_config = RouterConfig.config(model)

    if reasoning_effort:
        router_config["reasoning"] = {"effort": reasoning_effort}
        if "anthropic" in model.lower():
            router_config["reasoning"] = {
                "max_tokens": max_tokens // 1.5
            }
    
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

    sys_msg = messages[0]["content"] if messages else ""
    sys_summary = f"...{sys_msg[-1000:]}" if len(sys_msg) > 1000 else sys_msg

    logger.info(
        "[Agentic] Completion done",
        completion_message=choice.message,
        message_system_summary=sys_summary,
        messages_preview=messages[1:][-5:],  # last 5 non-system messages
        message_len=len(messages),
        max_tokens=max_tokens,
        model_completion=completion.model,
        model_requested=model,
        provider=completion.provider,
        runtime=runtime,
    )

    return completion

