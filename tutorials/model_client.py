#!/usr/bin/env python
# coding: utf-8

# We can directly use model client



from adalflow.components.model_client import OpenAIClient
from adalflow.core.types import ModelType
from adalflow.utils import setup_env

openai_client = OpenAIClient()

query = "What is the capital of France?"

# try LLM model
model_type = ModelType.LLM

prompt = f"User: {query}\n"
model_kwargs = {"model": "gpt-3.5-turbo", "temperature": 0.5, "max_tokens": 100}
api_kwargs = openai_client.convert_inputs_to_api_kwargs(input=prompt, 
                                                        model_kwargs=model_kwargs, 
                                                        model_type=model_type)
print(f"api_kwargs: {api_kwargs}")

response = openai_client.call(api_kwargs=api_kwargs, model_type=model_type)
response_text = openai_client.parse_chat_completion(response)
print(f"response_text: {response_text}")

# try embedding model
model_type = ModelType.EMBEDDER
# do batch embedding
input = [query] * 2
model_kwargs = {"model": "text-embedding-3-small", "dimensions": 8, "encoding_format": "float"}
api_kwargs = openai_client.convert_inputs_to_api_kwargs(input=input, model_kwargs=model_kwargs, model_type=model_type)
print(f"api_kwargs: {api_kwargs}")



response = openai_client.call(api_kwargs=api_kwargs, model_type=model_type)
reponse_embedder_output = openai_client.parse_embedding_response(response)
print(f"reponse_embedder_output: {reponse_embedder_output}")


# 
