#!/usr/bin/env python
# coding: utf-8

# ## Basic Generator Usage

# In this tutorial, we will show four ways to use generator:
# 1. Use directly by setting up a ``ModelClient``.
# 2. Customize prompt template using ``jinjia2``.
# 3. Try different models.
# 4. Use ``acall`` to do mutiple asynchronous calls for speed up.

# In default, the generator uses a default prompt template. It has these varaibles:
# 
# ```
# LIGHTRAG_DEFAULT_PROMPT_ARGS = [
#     "task_desc_str",
#     "output_format_str",
#     "tools_str",
#     "examples_str",
#     "chat_history_str",
#     "context_str",
#     "steps_str",
#     "input_str",
#     "output_str",
# ]
# ```



# first, let's set up the library log just in case, in default at INFO level
from adalflow.utils.logger import get_logger

get_logger()




from adalflow.core import Generator
from adalflow.components.model_client import OpenAIClient
from adalflow.utils import setup_env  # ensure you have .env with OPENAI_API_KEY

setup_env(".env")
query = "What is the capital of France?"
model_kwargs = {"model": "gpt-3.5-turbo"}
generator = Generator(model_client=OpenAIClient(), model_kwargs=model_kwargs)
prompt_kwargs = {
    "input_str": query,
}
# run the generator
output = generator(prompt_kwargs=prompt_kwargs)
print(output)


# The logging clearly shows us what we sent to OpenAI.



# lets see the prompt, it is quite minimal
generator.print_prompt(**prompt_kwargs)


# Writing your template is easy. Let us use our own template. Let's say, we want to set up our AI with a sense of humor.



template = """<SYS> Your are an assistant with a great sense of humor.</SYS> User: {{input_str}}. You:"""

generator2 = Generator(
    model_client=OpenAIClient(), model_kwargs=model_kwargs, template=template
)
response = generator2(prompt_kwargs=prompt_kwargs)
print(response)




# Let us use llama3 from groq
from adalflow.components.model_client import GroqAPIClient

groq_model_kwargs = {"model": "llama3-8b-8192"}
generator3 = Generator(
    model_client=GroqAPIClient(), model_kwargs=groq_model_kwargs, template=template
)

response = generator3(prompt_kwargs=prompt_kwargs)
print(response)




# Lets do 10 async calls at once, lets use GroqAPIClient
import nest_asyncio  # import asyncio, use nest_asyncio.apply() if you are in jupyter notebook
import asyncio

nest_asyncio.apply()

import time
from typing import List


async def make_async_calls(queries: List[str]):
    calls = [generator3.acall(prompt_kwargs={"input_str": query}) for query in queries]
    responses = await asyncio.gather(*calls)
    return responses


queries = [query] * 10
start = time.time()
responses = asyncio.run(make_async_calls(queries))
print(f"Time taken for 10 async calls: {time.time() - start}")
print(responses)






