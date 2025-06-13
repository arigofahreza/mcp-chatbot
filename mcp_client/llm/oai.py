import os
from typing import Optional

import dotenv
from dotenv import dotenv_values
from openai import OpenAI



dotenv.load_dotenv()


class OpenAIClient:
    def __init__(
        self,
        config
    ):
        self.embedding_model_name = config.llm_embedding_model_name,
        self.model_name = config.llm_model_name,
        self.client = OpenAI(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
        )

    def get_response(self, messages: list[dict[str, str]]) -> str:
        """Get a response from the Open AI LLM.

        Args:
            messages: A list of message dictionaries.

        Returns:
            The LLM's response as a string.
        """
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
        )
        return completion.choices[0].message.content

    def get_stream_response(
        self, messages: list[dict[str, str]]
    ):
        """Get a streaming response from the Open AI LLM.

        Args:
            messages: A list of message dictionaries.

        Yields:
            Chunks of the response as they arrive.
        """
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            stream=True,
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                yield content


    def get_embedding_response(self, message: str) -> dict:
        """Get an embedding response from the Open AI LLM.

        Args:
            message: A message dictionaries to embed with.

        Returns:
            The LLM's response as a dict json http response.
        """
        response = self.client.embeddings.create(
            input=message,
            model=self.embedding_model_name,
            dimensions=1024
        )
        return response.model_dump()
