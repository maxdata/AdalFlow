#!/usr/bin/env python
# coding: utf-8

# # Build a ChatBot 

# We have built a simple question-answering pipeline, where we can input a question and get an answer. In addition to single round of question-answering, we can also have a conversation with an LLM by building a chatbot. The chatbot can remember the history of the conversation and respond based on the history. The key to achieve this is to leverage the promt args `chat_history_str` and the data structure `Memory` to manage the conversation history.



# Import needed modules from LightRAG
from adalflow.core.component import Component
from adalflow.core.generator import Generator
from adalflow.core.memory import Memory




# Here, we use the OpenAIClient as an example, but you can use any other clients (with the corresponding API Key as needed), such as AnthropicAPIClient
from adalflow.components.model_client import OpenAIClient
OPENAI_API_KEY="YOUR_API_KEY" # Replace with your OpenAI API Key, or you can put it in a .env file




# Build the ChatBot pipeline
class ChatBot(Component):
    def __init__(self):
        super().__init__()
        self.generator = Generator(
            model_client=OpenAIClient(),
            model_kwargs={'model': 'gpt-3.5-turbo'}
        )
        self.chat_history = Memory() # Memory to store the chat history
    
    def call(self) -> str:
        print("Welcome to the ChatBot. Type anything to chat. Type 'exit' to end.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            chat_history_str = self.chat_history()
            # Generate the response from the user input and chat history
            response = self.generator(
                prompt_kwargs={
                    "input_str": user_input,
                    "chat_history_str": chat_history_str,
                },
            )
            # Save the user input and response to the memory
            self.chat_history.add_dialog_turn(
                user_query=user_input, assistant_response=response
            )
            print(f"ChatBot: {response}")

chatbot = ChatBot()
print(chatbot)




chatbot.call()

