#!/usr/bin/env python
# coding: utf-8

# <a target="_blank" href="https://colab.research.google.com/github.com/SylphAI-Inc/AdalFlow/blob/main/notebooks/tutorials/adalflow_dataclasses.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>
# 

# # 🤗 Welcome to AdalFlow!
# ## The library to build & auto-optimize any LLM task pipelines
# 
# Thanks for trying us out, we're here to provide you with the best LLM application development experience you can dream of 😊 any questions or concerns you may have, [come talk to us on discord,](https://discord.gg/ezzszrRZvT) we're always here to help! ⭐ <i>Star us on <a href="https://github.com/SylphAI-Inc/AdalFlow">Github</a> </i> ⭐
# 
# 
# # Quick Links
# 
# Github repo: https://github.com/SylphAI-Inc/AdalFlow
# 
# Full Tutorials: https://adalflow.sylph.ai/index.html#.
# 
# Deep dive on each API: check out the [developer notes](https://adalflow.sylph.ai/tutorials/index.html).
# 
# Common use cases along with the auto-optimization:  check out [Use cases](https://adalflow.sylph.ai/use_cases/index.html).
# 
# # Author
# 
# This notebook was created by community contributor [Ajith](https://github.com/ajithvcoder).
# 
# # Outline
# 
# This is a quick introduction of what AdalFlow is capable of. We will cover:
# 
# * How to use `DataClass` with `DataClassParser`.
# * How to do nested dataclass, we will test both one and two levels of nesting.
# 
# **Next: Try our [auto-optimization](https://colab.research.google.com/drive/1n3mHUWekTEYHiBdYBTw43TKlPN41A9za?usp=sharing)**
# 
# 
# # Installation
# 
# 1. Use `pip` to install the `adalflow` Python package. We will need `openai` and `groq`from the extra packages.
# 
#   ```bash
#   pip install adalflow[openai,groq]
#   ```
# 2. Setup  `openai` and `groq` API key in the environment variables

# ### Install adalflow



# Install adalflow with necessary dependencies
from IPython.display import clear_output

get_ipython().system('pip install -U adalflow[openai,groq]')

clear_output()


# ### Set Environment Variables
# 
# Note: Enter your api keys in below cell



get_ipython().run_cell_magic('writefile', '.env', '\nOPENAI_API_KEY="PASTE-OPENAI_API_KEY_HERE"\nGROQ_API_KEY="PASTE-GROQ_API_KEY-HERE"\n')




#  or more securely

import os

from getpass import getpass

# Prompt user to enter their API keys securely
groq_api_key = getpass("Please enter your GROQ API key: ")
openai_api_key = getpass("Please enter your OpenAI API key: ")


# Set environment variables
os.environ['GROQ_API_KEY'] = groq_api_key
os.environ['OPENAI_API_KEY'] = openai_api_key

print("API keys have been set.")


# ### Import necessary libraries



# Import required libraries
from dataclasses import dataclass, field
from typing import List, Dict
import adalflow as adal
from adalflow.components.model_client import GroqAPIClient
from adalflow.utils import setup_env




adal.__version__




# Load environment variables - Make sure to have OPENAI_API_KEY in .env file and .env is present in current folder
setup_env(".env")


# ### Basic Vannila Example



# Define the output structure using dataclass
@dataclass
class BasicQAOutput(adal.DataClass):
    explanation: str = field(
        metadata={"desc": "A brief explanation of the concept in one sentence."}
    )
    example: str = field(
        metadata={"desc": "An example of the concept in a sentence."}
    )
    # Control output fields order
    __output_fields__ = ["explanation", "example"]

# Define the template using jinja2 syntax
qa_template = r"""<SYS>
You are a helpful assistant.
<OUTPUT_FORMAT>
{{output_format_str}}
</OUTPUT_FORMAT>
</SYS>
<USER> {{input_str}} </USER>"""




# Define the QA component
class QA(adal.Component):
    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        super().__init__()

        # Initialize the parser with the output dataclass
        parser = adal.DataClassParser(data_class=BasicQAOutput, return_data_class=True)

        # Set up the generator with model, template, and parser
        self.generator = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=qa_template,
            prompt_kwargs={"output_format_str": parser.get_output_format_str()},
            output_processors=parser,
        )

    def call(self, query: str):
        """Synchronous call to generate response"""
        return self.generator.call({"input_str": query})

    async def acall(self, query: str):
        """Asynchronous call to generate response"""
        return await self.generator.acall({"input_str": query})




# Example usage
def run_basic_example():
    # Instantiate the QA class with Groq model
    qa = QA(
        model_client=GroqAPIClient(),
        model_kwargs={"model": "llama3-8b-8192"},
    )

    # Print the QA instance details
    print(qa)

    # Test the QA system
    response = qa("What is LLM?")
    print("\nResponse:")
    print(response)
    print(f"BasicQAOutput: {response.data}")
    print(f"Explanation: {response.data.explanation}")
    print(f"Example: {response.data.example}")




run_basic_example()


# ### Example 1 - Movie analysis data class



# 1. Basic DataClass with different field types
@dataclass
class MovieReview(adal.DataClass):
    title: str = field(
        metadata={"desc": "The title of the movie"}
    )
    rating: float = field(
        metadata={
            "desc": "Rating from 1.0 to 10.0",
            "min": 1.0,
            "max": 10.0
        }
    )
    pros: List[str] = field(
        default_factory=list,
        metadata={"desc": "List of positive points about the movie"}
    )
    cons: List[str] = field(
        default_factory=list,
        metadata={"desc": "List of negative points about the movie"}
    )

    __output_fields__ = ["title", "rating", "pros", "cons"]




@dataclass
class Actor(adal.DataClass):
    name: str = field(metadata={"desc": "Actor's full name"})
    role: str = field(metadata={"desc": "Character name in the movie"})




# 2. Nested DataClass example

# Have both MovieReview and Actor nested in DetailedMovieReview

@dataclass
class DetailedMovieReview(adal.DataClass):
    basic_review: MovieReview
    cast: List[Actor] = field(
        default_factory=list,
        metadata={"desc": "List of main actors in the movie"}
    )
    genre: List[str] = field(
        default_factory=list,
        metadata={"desc": "List of genres for the movie"}
    )
    recommend: bool = field(
        default_factory=str,
        metadata={"desc": "Whether you would recommend this movie"}
    )

    __output_fields__ = ["basic_review", "cast", "genre", "recommend"]




# Example template for movie review
movie_review_template = r"""<SYS>
You are a professional movie critic. Analyze the given movie and provide a detailed review.
<OUTPUT_FORMAT>
{{output_format_str}}
</OUTPUT_FORMAT>
</SYS>
<USER> Review this movie: {{movie_title}} </USER>"""




# Create the MovieReviewer component with MovieAnalysis data class
class MovieReviewer(adal.Component):
    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict, data_class: adal.DataClass):
        super().__init__()
        self.additional_structure_prompt = "Dont use 'type' and 'properties' in output directly give as dict"
        parser = adal.DataClassParser(
            data_class=data_class,
            return_data_class=True
        )
        self.generator = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=movie_review_template,
            prompt_kwargs={"output_format_str": parser.get_output_format_str() + self.additional_structure_prompt},
            output_processors=parser,
        )

    def call(self, movie_title: str):
        return self.generator.call({"movie_title": movie_title})




# test the data class with one level of nesting

reviewer = MovieReviewer(
    model_client=GroqAPIClient(),
    model_kwargs={"model": "llama3-8b-8192"},
    data_class=DetailedMovieReview
)

response = reviewer("The Matrix")
print(f"DetailedMovieReview: {response.data}")
print(f"BasicReview: {response.data.basic_review}")
print(f"Cast: {response.data.cast}")




# try use openai model
reviewer = MovieReviewer(
    model_client=adal.OpenAIClient(),
    model_kwargs={"model": "gpt-4o"},
    data_class=DetailedMovieReview
)
response = reviewer("The Matrix")
print(f"DetailedMovieReview: {response.data}")
print(f"BasicReview: {response.data.basic_review}")
print(f"Cast: {response.data.cast}")


# We see both models can handle one level of nested dataclass quite well. And the output ordering will follow the ordering specified in __output_fields__



# 3. second level nested dataclass

@dataclass
class MovieAnalysis(adal.DataClass):
    review: DetailedMovieReview
    box_office: float = field(
        default=None,
        metadata={"desc": "Box office earnings in millions of dollars"}
    )
    awards: Dict[str, int] = field(
        default=None,
        metadata={"desc": "Dictionary of award categories and number of wins"}
    )

    __output_fields__ = ["review", "box_office", "awards"]




# test the data class with two levels of nested dataclass

# gpt-3.5-turbo model

analysis = MovieReviewer(
    model_client=adal.OpenAIClient(),
    model_kwargs={"model": "gpt-3.5-turbo"},
    data_class=MovieAnalysis
)

response = analysis("The Matrix")
print(f"MovieAnalysis: {response.data}")
print(f"DetailedMovieReview: {response.data.review}")
print(f"BasicReview: {response.data.review.basic_review}")
print(f"Cast: {response.data.review.cast}")




# test the data class with two levels of nested dataclass

analysis = MovieReviewer(
    model_client=GroqAPIClient(),
    model_kwargs={"model": "llama3-8b-8192"},
    data_class=MovieAnalysis
)

response = analysis("The Matrix")
print(f"MovieAnalysis: {response.data}")
print(f"DetailedMovieReview: {response.data.review}")
print(f"BasicReview: {response.data.review.basic_review}")
print(f"Cast: {response.data.review.cast}")


# ### Example 2: Song Review
# Note: Song Review is modified by keeping Example 1 - Movie Review as a reference so that we would know how to use DataClasses for similar purposes



# 1. Basic DataClass with different field types
@dataclass
class SongReview(adal.DataClass):
    title: str = field(
        metadata={"desc": "The title of the song"}
    )
    album: str = field(
        metadata={"desc": "The album of the song"}
    )
    ranking: int = field(
        metadata={
            "desc": "Billboard peak ranking from 1 to 200",
            "min": 1,
            "max": 200
        }
    )
    streaming: Dict[str, int] = field(
        default_factory=list,
        metadata={"desc": "Dict of lastest approximate streaming count in spotify and in youtube. Gives the count in millions"}
    )
    pros: List[str] = field(
        default_factory=list,
        metadata={"desc": "List of positive points about the song"}
    )
    cons: List[str] = field(
        default_factory=list,
        metadata={"desc": "List of negative points about the song"}
    )

    __output_fields__ = ["title", "rating", "streaming", "pros", "cons"]




@dataclass
class Artist(adal.DataClass):
    name: str = field(metadata={"desc": "Artist's full name"})
    role: str = field(metadata={"desc": "Artist's role in the song"})




# 2. Nested DataClass example

@dataclass
class DetailedSongReview(adal.DataClass):
    basic_review: SongReview = field(
        default=SongReview, metadata={"desc": "basic Song review details"}
    )
    cast: List[Artist] = field(
        default_factory=list,
        metadata={"desc": "List of main singer, lyrisist and musicians in the song"}
    )
    genre: List[str] = field(
        default_factory=list,
        metadata={"desc": "List of genres for the song"}
    )
    recommend: bool = field(
        default_factory=str,
        metadata={"desc": "Whether you would recommend this song"}
    )

    __output_fields__ = ["basic_review", "cast", "genre", "recommend"]




# 3. two levels of nesting dataclass

# all these fields as we use default, it is optional, so 
# llm might not output that field if they dont have information

@dataclass
class SongAnalysis(adal.DataClass):
    review: DetailedSongReview = field(
        default=DetailedSongReview, metadata={"desc": "Song review details"}
    )
    duration: float = field(
        default=None,
        metadata={"desc": "Duration of the song"}
    )
    awards: Dict[str, int] = field(
        default=None,
        metadata={"desc": "Dictionary of award categories and number of wins"}
    )

    __output_fields__ = ["review", "duration", "awards"]




# Example template for song review
song_review_template = r"""<SYS>
You are a professional song critic. Analyze the given song and provide a detailed review.
<OUTPUT_FORMAT>
{{output_format_str}}
</OUTPUT_FORMAT>
</SYS>
<USER> Review this song: {{song_title}} </USER>"""




# Create the SongReviewer component with SongAnalysis data class
class SongReviewer(adal.Component):
    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        super().__init__()
        self.additional_structure_prompt = "Dont use 'type' and 'properties' in output directly give as dict"
        parser = adal.DataClassParser(
            data_class=SongAnalysis,
            return_data_class=False,
            format_type="json"
        )
        self.generator = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=song_review_template,
            prompt_kwargs={"output_format_str": parser.get_output_format_str() + self.additional_structure_prompt },
            output_processors=parser,
        )

    def call(self, song_title: str):
        return self.generator.call({"song_title": song_title})




analysis = SongReviewer(
     model_client=GroqAPIClient(),
     model_kwargs={"model": "llama3-8b-8192"},
)

response = analysis("Shape of you")
print(f"SongAnalysis: {response.data}")

# this time as we set `return_data_class` to False in the parser, we get the output as dict




# Access nested data
analysis = response.data
print(f"Song Title: {analysis['review']['basic_review']['title']}")
print(f"Album: {analysis['review']['basic_review']['album']}")
print(f"Ranking: {analysis['review']['basic_review']['ranking']}")

for platform, views in analysis['review']['basic_review']['streaming'].items():
    print(f"- {platform} - {views} million views")
print("\nPros:")
for pro in analysis['review']["basic_review"]["pros"]:
    print(f"- {pro}")

print("\nArtist's:")
for actor in analysis['review']["cast"]:
        print(f"- {actor['name']} as {actor['role']}")

if analysis['review']['genre']:
    print(f"\nGenere:  ")
    for genre in analysis['review']['genre']:
        print(f" {genre} ")

if analysis['duration']:
    print(f"\nDuration: {analysis['duration']} minutes")

if hasattr(analysis, 'awards') and analysis['awards']:
    print("\nAwards:")
    for category, count in analysis['awards'].items():
        print(f"- {category}: {count}")


# TODOs:
# 1. Add `JsonOutputParser` and `YamlOutputParser` to this notebook.

# # Issues and feedback
# 
# If you encounter any issues, please report them here: [GitHub Issues](https://github.com/SylphAI-Inc/LightRAG/issues).
# 
# For feedback, you can use either the [GitHub discussions](https://github.com/SylphAI-Inc/LightRAG/discussions) or [Discord](https://discord.gg/ezzszrRZvT).
