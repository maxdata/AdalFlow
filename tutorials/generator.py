#!/usr/bin/env python
# coding: utf-8



from adalflow.core import Generator
from adalflow.components.model_client import OpenAIClient, get_all_messages_content, get_probabilities
from adalflow.utils import enable_library_logging

enable_library_logging(level="DEBUG")

model_kwargs={
    "model": "gpt-3.5-turbo",
    "logprobs": True,
    "n": 2, # the number of chat completion choices
}
model_client = OpenAIClient(chat_completion_parser=get_probabilities)
generator = Generator(model_client=model_client, model_kwargs=model_kwargs)


prompt_kwargs = {
    "input_str": "What is the capital of France?",
}
output = generator(prompt_kwargs=prompt_kwargs)
print(output)


# 2024-06-19 09:14:19 - openai_client - DEBUG - [openai_client.py:81:parse_chat_completion] - completion: ChatCompletion(id='chatcmpl-9bsEtR7IkwwKKuXao3hCFx210vThx', choices=[Choice(finish_reason='stop', index=0, logprobs=ChoiceLogprobs(content=[ChatCompletionTokenLogprob(token='The', bytes=[84, 104, 101], logprob=-0.29857525, top_logprobs=[]), ChatCompletionTokenLogprob(token=' capital', bytes=[32, 99, 97, 112, 105, 116, 97, 108], logprob=-4.604148e-06, top_logprobs=[]), ChatCompletionTokenLogprob(token=' of', bytes=[32, 111, 102], logprob=-0.00016754455, top_logprobs=[]), ChatCompletionTokenLogprob(token=' France', bytes=[32, 70, 114, 97, 110, 99, 101], logprob=-3.0545007e-06, top_logprobs=[]), ChatCompletionTokenLogprob(token=' is', bytes=[32, 105, 115], logprob=-2.220075e-06, top_logprobs=[]), ChatCompletionTokenLogprob(token=' Paris', bytes=[32, 80, 97, 114, 105, 115], logprob=-4.00813e-06, top_logprobs=[]), ChatCompletionTokenLogprob(token='.', bytes=[46], logprob=-0.0001039008, top_logprobs=[])]), message=ChatCompletionMessage(content='The capital of France is Paris.', role='assistant', function_call=None, tool_calls=None)), Choice(finish_reason='stop', index=1, logprobs=ChoiceLogprobs(content=[ChatCompletionTokenLogprob(token='Paris', bytes=[80, 97, 114, 105, 115], logprob=-1.3551816, top_logprobs=[])]), message=ChatCompletionMessage(content='Paris', role='assistant', function_call=None, tool_calls=None))], created=1718813659, model='gpt-3.5-turbo-0125', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=8, prompt_tokens=22, total_tokens=30))
# 



from adalflow.core import Component, Generator, Prompt
from adalflow.components.model_client import GroqAPIClient
from adalflow.utils import setup_env # noqa


class SimpleQA(Component):
    def __init__(self):
        super().__init__()
        template = r"""<SYS>
        You are a helpful assistant.
        </SYS>
        User: {{input_str}}
        You:
        """
        self.generator = Generator(
            model_client=GroqAPIClient(), model_kwargs={"model": "llama3-8b-8192"}, template=template
        )

    def call(self, query):
        return self.generator({"input_str": query})

    async def acall(self, query):
        return await self.generator.acall({"input_str": query})


qa = SimpleQA()
answer = qa("What is AdalFLow?")

print(answer)

