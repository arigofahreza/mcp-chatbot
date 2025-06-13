import os
from typing import Optional

import dotenv
import requests
from dotenv import dotenv_values

dotenv.load_dotenv()


class OllamaClient:
    def __init__(
            self,
            config
    ):
        self.embedding_model_name = config.ollama_embedding_model_name
        self.model_name = config.ollama_model_name
        self.api_base = config.ollama_base_url

    def get_response(self, messages: list[dict[str, str]]) -> str:
        """Get a response from the Ollama LLM.

        Args:
            messages: A list of message dictionaries with 'role' and 'content'.

        Returns:
            The LLM's response as a string.
        """
        # Ollama API expects a specific format for the request
        response = requests.post(
            f"{self.api_base}/api/chat",
            json={
                "model": self.model_name,
                "messages": messages,
                "stream": False,
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def get_stream_response(self, messages: list[dict[str, str]]):
        """Get a streaming response from the Ollama LLM.

        Args:
            messages: A list of message dictionaries.

        Yields:
            Chunks of the response as they arrive.
        """
        # Use requests to stream the response
        response = requests.post(
            f"{self.api_base}/api/chat",
            json={
                "model": self.model_name,
                "messages": messages,
                "stream": True,
            },
            stream=True,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8")
                # Skip "done" message
                if data == '{"done":true}':
                    continue

                # Parse the JSON response
                import json

                try:
                    chunk = json.loads(data)
                    if "message" in chunk and "content" in chunk["message"]:
                        content = chunk["message"]["content"]
                        if content:
                            yield content
                except json.JSONDecodeError:
                    continue

    def get_embedding_response(self, message: str) -> dict:
        """Get an embedding response from the Ollama LLM.

        Args:
            message: A message dictionaries to embed with.

        Returns:
            The LLM's response as a dict json http response.
        """
        headers = {
            'Content-Type': 'application/json',
        }

        response = requests.post(
            f"{self.api_base}/api/embed",
            json={
                'model': self.embedding_model_name,
                'input': message
            },
            headers=headers
        )
        response.raise_for_status()
        return response.json()
