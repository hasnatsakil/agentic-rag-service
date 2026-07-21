"""
OpenRouter client and model configuration.

This module initialises the OpenAI-compatible client pointed at the
OpenRouter API, and defines three sets of enumerations for model selection:

- :class:`RouterModel`         — primary LLMs for answer generation.
- :class:`RouterEmbeddingModel`— embedding models for chunk and query vectorisation.
- :class:`RouterGradingModel`  — lightweight models for binary relevance grading.

:class:`RouterConfig` aggregates these into ordered priority lists and
exposes a :meth:`~RouterConfig.config` factory that builds the
``extra_body`` dict consumed by the OpenRouter ``/chat/completions`` endpoint.

Environment variables:
    OPENROUTER_API_KEY: OpenRouter API key, loaded from ``.env`` via
        :mod:`dotenv`.
"""

import os
from enum import Enum
from typing import Optional

import structlog
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = structlog.get_logger(__name__)

#: OpenAI-compatible client configured to use the OpenRouter base URL.
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),  # type: ignore[arg-type]
)


class RouterModel(Enum):
    """Available LLMs for answer generation, routed through OpenRouter.

    Members prefixed with free-tier models (no cost per token); paid models
    offer higher quality or throughput.
    """

    # -- Free-tier models --------------------------------------------------
    MINIMAX25 = "minimax/minimax-m2.5:free"
    QWEN3NEXT = "qwen/qwen3-next-80b-a3b-instruct:free"
    GPTOSS    = "openai/gpt-oss-20b:free"
    COBUDDY   = "baidu/cobuddy:free"

    # -- Paid models -------------------------------------------------------
    STEP_3_5_FLASH      = "stepfun/step-3.5-flash"
    TENCENT_HY3_PREVIEW = "tencent/hy3-preview"
    TRINITY_LARGE       = "arcee-ai/trinity-large-thinking"


class RouterEmbeddingModel(Enum):
    """OpenAI embedding models available through OpenRouter.

    ``SMALL`` is used as the primary embedding model due to its lower cost
    and sufficient quality for most RAG workloads.
    """

    SMALL = "openai/text-embedding-3-small"
    LARGE = "openai/text-embedding-3-large"


class RouterGradingModel(Enum):
    """Lightweight LLMs used for binary relevance and hallucination grading.

    These models are intentionally small to minimise latency and cost for
    yes/no classification tasks (``max_tokens=5``).
    """

    NEMOTRON   = "nvidia/nemotron-nano-9b-v2:free"
    NORTH_MINI = "cohere/north-mini-code:free"
    LLAMA      = "meta-llama/llama-3-8b-instruct:free"


class RouterConfig:
    """Factory class for OpenRouter provider configuration dicts.

    Holds ordered priority lists for providers and models, and exposes
    :meth:`config` to build the ``extra_body`` payload that OpenRouter uses
    to apply fallback routing, quantisation filters, and optional plugins.

    Class Attributes:
        FALLBACK (bool): Whether to allow provider fallbacks. Default ``True``.
        QUANTIZATION (list): Quantisation constraints (empty = any).
        PROVIDERS_IGNORED (list): Provider IDs to exclude from routing.
        PROVIDERS_PRIORITY (list): Ordered list of preferred providers.
        MODELS_PRIORITY (list): Fallback model chain for standard completions.
        PAID_MODELS_PRIORITY (list): Fallback chain for paid-tier completions.
        EMBEDDING_MODELS_PRIORITY (list): Fallback chain for embedding calls.
        GRADEING_MODELS_PRIORITY (list): Fallback chain for grading calls.
    """

    FALLBACK: bool = True
    QUANTIZATION: list = []
    PROVIDERS_IGNORED: list = []
    PROVIDERS_PRIORITY: list = []

    MODELS_PRIORITY: list = [
        RouterModel.QWEN3NEXT.value,
        RouterModel.GPTOSS.value,
        RouterModel.COBUDDY.value,
    ]
    PAID_MODELS_PRIORITY: list = [
        RouterModel.STEP_3_5_FLASH.value,
        RouterModel.TENCENT_HY3_PREVIEW.value,
        RouterModel.TRINITY_LARGE.value,
    ]
    EMBEDDING_MODELS_PRIORITY: list = [
        RouterEmbeddingModel.SMALL.value,
        RouterEmbeddingModel.LARGE.value,
    ]
    GRADEING_MODELS_PRIORITY: list = [
        RouterGradingModel.NEMOTRON.value,
        RouterGradingModel.NORTH_MINI.value,
        RouterGradingModel.LLAMA.value,
    ]

    @classmethod
    def config(
        cls,
        MODEL: str = RouterModel.TRINITY_LARGE.value,
        search_prompt: Optional[str] = None,
        has_image: bool = False,
        is_grading: bool = False,
        is_hallucination: bool = False,
    ) -> dict:
        """Build the OpenRouter ``extra_body`` configuration dict.

        Constructs provider routing settings, selects the appropriate model
        fallback list, and optionally activates the web-search plugin.

        Args:
            MODEL: The primary model identifier. Used as a hint for
                provider selection.
            search_prompt: When provided, enables the OpenRouter web-search
                plugin with the given prompt and ``exa`` as the search engine.
            has_image: Reserved for future multimodal support. Currently
                unused.
            is_grading: When ``True``, uses :attr:`GRADEING_MODELS_PRIORITY`
                as the model fallback list.
            is_hallucination: Reserved for hallucination-specific routing.
                Currently falls through to the standard model list.

        Returns:
            A dict suitable for passing as ``extra_body`` to
            ``client.chat.completions.create()``.  Includes ``"provider"``,
            ``"models"``, and ``"route"`` keys, plus an optional
            ``"plugins"`` key when ``search_prompt`` is set.
        """
        configuration = {
            "provider": {
                "order": cls.PROVIDERS_PRIORITY,
                "ignore": cls.PROVIDERS_IGNORED,
                "allow_fallbacks": cls.FALLBACK,
                "quantizations": cls.QUANTIZATION,
            },
            "models": (
                cls.GRADEING_MODELS_PRIORITY
                if (is_grading or is_hallucination)
                else ( cls.PAID_MODELS_PRIORITY[:2])
            ),
            "route": "fallback",
        }

        if search_prompt:
            configuration["plugins"] = [
                {
                    "id": "web",
                    "engine": "exa",         # Options: "native", "exa", or omit for default.
                    "max_results": 5,
                    "search_prompt": search_prompt,
                }
            ]

        return configuration
