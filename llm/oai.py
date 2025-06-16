import os
from typing import Optional

import dotenv
from openai import OpenAI

from helpers.generator import serialize_f32

dotenv.load_dotenv()


class OpenAIClient:
    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.embedding_model_name = embedding_model_name or os.getenv('LLM_EMBEDDING_MODEL_NAME')
        self.model_name = model_name or os.getenv("LLM_MODEL_NAME")
        self.client = OpenAI(
            api_key=api_key or os.getenv("LLM_API_KEY"),
            base_url=base_url or os.getenv("LLM_BASE_URL"),
        )

    def get_response(self, messages: list[dict[str, str]]) -> str:
        """Get a response from the LLM.

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
        """Get a streaming response from the LLM.

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

    def get_embedding_response(self, datas: list[str]) -> list:
        """Get a response from the embedding LLM.

        Args:
            datas: list of sentence to embed.

        Returns:
            The LLM's embedding response as a list.
        """
        response = self.client.embeddings.create(
            model=self.embedding_model_name,
            input=datas,
            dimensions=1024
        )
        data = []
        for record in response.data:
            data.append((record.index+1, serialize_f32(record.embedding)))
        return data



if __name__ == "__main__":
    client = OpenAIClient()
    # Testing.
    print(client.get_response([{"role": "user", "content": "你是谁？"}]))

    # Testing stream response
    for chunk in client.get_stream_response([{"role": "user", "content": "你是谁？"}]):
        print(chunk, end="", flush=True)
