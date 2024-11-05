#!/usr/bin/env python
# coding: utf-8

# Our task is to build a consultant bot that can answer questions of different domains, such as medical with a doctor bot or legal with a lawyer bot.
# 
# We will show how flexible ``Component`` and the ``Sequential`` container is to build the same task
# in different ways.
# 
# 1. **Single Task**: We can build a single task where it deals with multiple generators and handles any coding logic.
# 2. **Multiple Tasks** and combine them using ``Sequential`` which resembles the concept of `Chain` or pipelines in other libraries.

# First, lets prepare the imports and prompt templates using `jinjia2` template. We plan to demonstrate how we can use different models too. If this tutorial is the first thing you read, no need to care about more details, but focus on how the `development process` looks like using `LightRAG` library.



import re
from adalflow.core import Component, Generator, Sequential
from adalflow.components.model_client import OpenAIClient
from adalflow.components.model_client import GroqAPIClient
from adalflow.utils import setup_env # make sure you have a .env file with OPENAI_API_KEY and GROQ_API_KEY
setup_env(".env")




template_doc = r"""<SYS> You are a doctor </SYS> User: {{input_str}}"""
template_law = r"""<SYS> You are a lawyer </SYS> User: {{input_str}}"""
template_router = r"""<SYS> You are a router who will route a user question to the right generator.
            Here are your choices in form of key: value pairs:
             {% for key, value in choices.items() %}
                {{ key }}: {{ value }}
             {% endfor %}
            Output the key of your choice.
            </SYS> User question: {{input_str}}
            You:
            """




# Let's turn on the library log to help with debugging.
from adalflow.utils import get_logger
get_logger()


# Here is our first approach to build a single task with multiple generators and call each conditionally.



class ChatBotWithRouter(Component):
    def __init__(self):
        super().__init__()
        model_1_kwargs = {
            "model": "gpt-3.5-turbo",
        }
        model_2_kwargs = {"model": "llama3-8b-8192"}
        self.doc = Generator(
            template=template_doc,
            model_client=OpenAIClient(),
            model_kwargs=model_1_kwargs,
        )
        self.lawyer = Generator(
            template=template_law,
            model_client=GroqAPIClient(),
            model_kwargs=model_2_kwargs,
        )
        self.router_choices = {
            "doctor": self.create_generator_signature(self.doc),
            "lawyer": self.create_generator_signature(self.lawyer),
            "other": "Choose me the question does not apply to other choices.",
        }
        print(self.router_choices)

        self.router = Generator(
            template=template_router,
            model_client=OpenAIClient(),
            model_kwargs=model_1_kwargs,
        )

    def call(self, query: str) -> str:
        choice = self.router(
            prompt_kwargs={"input_str": query, "choices": self.router_choices}
        ).data
        if choice == "doctor":
            return self.doc(prompt_kwargs={"input_str": query}).data
        elif choice == "lawyer":
            return self.lawyer(prompt_kwargs={"input_str": query}).data
        else:
            return "Sorry, I cannot help you with that."

    def create_generator_signature(self, generator: Generator):
        template = generator.template
        pattern = r"<SYS>(.*?)</SYS>"

        matches = re.findall(pattern, template)
        for match in matches:
            print("Content between <SYS> tags:", match)
            return match




# Initiate the task component, and print the task details.

task = ChatBotWithRouter()
task




# Call the task with a query

query = "I have a legal question"
print(task(query))


# Now, let's separate this into multiple subtasks and ``chain`` them using the ``Sequential`` container.
# 
# First, the router task which will takes a dictionary of choices and return the selected key. In addition, we use ``_extra_repr`` to improve the default string representation of the task.
# 
# As ``Sequential`` will pass the output of one task to the next using positional arguments, we return whatever is needed to the next task in a dictionary.



# Router component

from typing import Dict
class Router(Component):
    def __init__(self, choices: Dict[str, str] = {}):
        super().__init__()
        self.choices = choices
        self.router = Generator(
            template=template_router,
            model_client=OpenAIClient(),
            model_kwargs={"model": "gpt-3.5-turbo"},
        )

    def call(self, query: str) -> str:
        prompt_kwargs = {"input_str": query, "choices": self.choices}
        choice =  self.router(prompt_kwargs=prompt_kwargs).data
        return {"choice": choice, "query": query}
    
    def _extra_repr(self):
        return f"Choices: {self.choices}, "




r = Router()
r


# Now, lets build another subtask which handles the chat depending on the selected key from the router task.
# As the router task returns a dictionary, we will make our input dictionary type that parses the ``choice`` and ``query`` key value pairs.



# the second chat component with two generators

class Chat(Component):
    def __init__(self):
        super().__init__()
        self.doc = Generator(
            template=template_doc,
            model_client=OpenAIClient(),
            model_kwargs={"model": "gpt-3.5-turbo"},
        )
        self.lawyer = Generator(
            template=template_law,
            model_client=GroqAPIClient(),
            model_kwargs={"model": "llama3-8b-8192"},
        )
    # to chain together just to make sure the output can be directly passed to the next as input
    def call(self, input: Dict[str, str]) -> Dict[str, str]:
        choice = input.get("choice", None)
        query = input.get("query", None)
        if choice == "doctor":
            return self.doc(prompt_kwargs={"input_str": query}).data
        elif choice == "lawyer":
            return self.lawyer(prompt_kwargs={"input_str": query}).data
        else:
            return "Sorry, I am not able to help you with that."




chat = Chat()
chat


# Now, lets chain the router and the chat task using the ``Sequential`` container into a runnable pipeline.



class QAWithRouter(Component):
    def __init__(self):
        super().__init__()
        self.router = Router(choices={"doctor": "Doctor", "lawyer": "Lawyer", "other": "Other"})
        self.chat = Chat()
        self.pipeline = Sequential(self.router, self.chat)

    def call(self, query: str) -> str:
        return self.pipeline(query)




qa = QAWithRouter()
qa




qa("I have a legal question")

