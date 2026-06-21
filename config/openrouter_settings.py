import os
from enum import Enum

import structlog
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = structlog.get_logger(__name__)


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),  #type: ignore
)


class RouterModel(Enum):
    # FREE MODELS
    MINIMAX25 = "minimax/minimax-m2.5:free"
    QWEN3NEXT = "qwen/qwen3-next-80b-a3b-instruct:free"
    GPTOSS = "openai/gpt-oss-20b:free"
    COBUDDY = "baidu/cobuddy:free"
    # PAID MODELS
    STEP_3_5_FLASH = "stepfun/step-3.5-flash"
    TENCENT_HY3_PREVIEW = "tencent/hy3-preview"
    TRINITY_LARGE = "arcee-ai/trinity-large-thinking"

class RouterEmbeddingModel(Enum):
    SMALL = "openai/text-embedding-3-small"
    LARGE = "openai/text-embedding-3-large"


class RouterConfig:
    FALLBACK = True
    QUANTIZATION = []
    PROVIDERS_IGNORED = []
    PROVIDERS_PRIORITY = []
    MODELS_PRIORITY = [
        RouterModel.QWEN3NEXT.value,
        RouterModel.GPTOSS.value,
        RouterModel.COBUDDY.value,
    ]
    # MODELS_PRIORITY = [
    #     RouterModel.STEP_3_5_FLASH.value,
    #     RouterModel.TENCENT_HY3_PREVIEW.value,
    #     RouterModel.TRINITY_LARGE.value,
    # ]

    @classmethod
    def config(
        cls,
        MODEL: str = RouterModel.TRINITY_LARGE.value,
        search_prompt: str = None,
        has_image: bool = False,
    ):

        configartion = {
            "provider": {
                "order": cls.PROVIDERS_PRIORITY,
                "ignore": cls.PROVIDERS_IGNORED,
                "allow_fallbacks": cls.FALLBACK,
                "quantizations": cls.QUANTIZATION,
            },
            "models": cls.MODELS_PRIORITY,
            "route": "fallback",
        }
        if search_prompt:
            configartion["plugins"] = [
                {
                    "id": "web",
                    "engine": "exa",  # Optional: "native", "exa", or undefined
                    "max_results": 5,
                    "search_prompt": search_prompt,
                }
            ]
        return configartion
