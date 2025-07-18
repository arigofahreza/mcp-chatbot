from typing import Literal

from config.configuration import Configuration
from llm.oai import OpenAIClient
from llm.ollama import OllamaClient

LLMProvider = Literal["openai", "ollama"]


def create_llm_client(
    provider: LLMProvider, config: Configuration
) -> OpenAIClient | OllamaClient:
    """Create appropriate LLM client based on provider.

    Args:
        provider: LLM provider type ("openai" or "ollama")
        config: Configuration object containing LLM model name, API key, and base URL

    Returns:
        Initialized LLM client instance
    """
    if provider == "openai":
        return OpenAIClient(
            embedding_model_name=config.llm_embedding_model_name,
            model_name=config.llm_model_name,
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
        )
    elif provider == "ollama":
        return OllamaClient(
            embedding_mode_name=config.ollama_embedding_model_name,
            model_name=config.ollama_model_name,
            api_base=config.ollama_base_url,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
