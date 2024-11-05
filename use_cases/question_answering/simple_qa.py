#!/usr/bin/env python
# coding: utf-8

# # Build A Simple Question-Answering Pipeline

# In this use case, we show how to build a simple question-answering pipeline.



# Import needed modules from LightRAG
from adalflow.core.component import Component
from adalflow.core.generator import Generator




# Here, we use the OpenAIClient as an example, but you can use any other clients (with the corresponding API Key as needed), such as AnthropicAPIClient
from adalflow.utils import setup_env # make sure you have a .env file with OPENAI_API_KEY or any other key mentioned with respect to your usage
setup_env(".env")
from adalflow.components.model_client import OpenAIClient




# Build the SimpleQA pipeline
class SimpleQA(Component):
    def __init__(self):
        super().__init__()
        self.generator = Generator(
            model_client=OpenAIClient(),
            model_kwargs={'model': 'gpt-3.5-turbo'}
        )

    def call(self, query: str):
        return self.generator.call(prompt_kwargs={'input_str': query})

simple_qa = SimpleQA()
print(simple_qa)




query = "What is the capital of France?"
response = simple_qa.call(query)
print(response)

