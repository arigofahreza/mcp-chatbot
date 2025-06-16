import os
from typing import Optional

import dotenv
import requests

from helpers.generator import serialize_f32

dotenv.load_dotenv()


class OllamaClient:
    def __init__(
            self,
            embedding_mode_name: Optional[str] = None,
            model_name: Optional[str] = None,
            api_base: Optional[str] = None,
    ):
        self.embedding_model_name = embedding_mode_name or os.getenv('OLLAMA_EMBEDDING_MODEL_NAME')
        self.model_name = model_name or os.getenv("OLLAMA_MODEL_NAME")
        self.api_base = api_base or os.getenv(
            "OLLAMA_API_BASE", "http://localhost:11434"
        )

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

    def get_embedding_response(self, datas: list[str]) -> list:
        """Get a response from the Ollama Embedding LLM.

        Args:
            datas: list of sentence to embed.

        Returns:
            The Ollama LLM's embedding response as a list.
        """
        # Ollama API expects a specific format for the request
        response = requests.post(
            f"{self.api_base}/api/embed",
            json={
                "model": self.embedding_model_name,
                "input": datas,
            },
        )
        response.raise_for_status()
        data = []
        for index, record in enumerate(response.json().get('embeddings')):
            data.append((index+1, serialize_f32(record)))
        return data


if __name__ == "__main__":
    client = OllamaClient(
        model_name="deepseek-r1:32b", api_base="http://localhost:11434"
    )
    # Testing
    print(client.get_response([{"role": "user", "content": "你是谁？"}]))

    # Testing stream response
    print("\nStreaming response:")
    for chunk in client.get_stream_response([{"role": "user", "content": "你是谁？"}]):
        print(chunk, end="", flush=True)
