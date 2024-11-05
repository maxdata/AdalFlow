#!/usr/bin/env python
# coding: utf-8



from openai import OpenAI
import json
from adalflow.utils import setup_env

client = OpenAI()

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

def run_conversation():
    # Step 1: send the conversation and available functions to the model
    messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    print(response)
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_current_weather": get_current_weather,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        second_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )  # get a new response from the model where it can see the function response
        return second_response
print(run_conversation())




from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
import time
import asyncio

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    time.sleep(1)
    return a * b

def add(a: int, b: int) -> int:
    """Add two numbers."""
    time.sleep(1)
    return a + b

async def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    await asyncio.sleep(1)
    return float(a) / b


async def search(query: str) -> List[str]:
    """Search for query and return a list of results."""
    await asyncio.sleep(1)
    return ["result1" + query, "result2" + query]


def numpy_sum(arr: np.ndarray) -> float:
    """Sum the elements of an array."""
    return np.sum(arr)

x = 2
@dataclass
class Point:
    x: int
    y: int

def add_points(p1: Point, p2: Point) -> Point:
    return Point(p1.x + p2.x, p1.y + p2.y)

all_functions = [multiply, add, divide, search, numpy_sum, add_points]

all_functions_dict = {f.__name__: f for f in all_functions}




# describing the functions

from adalflow.core.func_tool import FunctionTool

functions =[multiply, add, divide, search, numpy_sum, add_points]
tools = [
    FunctionTool(fn=fn) for fn in functions
]
for tool in tools:
    print(tool)




# create a context map
context_map = {tool.definition.func_name: tool for tool in tools}




print(tools[-2].definition.to_dict())

print(tools[-2].definition.to_json())

print(repr(tools[-2].definition.to_yaml()))




# tool definition for get_current_weather

ft = FunctionTool(fn=get_current_weather)
ft.definition.to_dict()




# to further help us manage the whole process, we will use a tool manager

from adalflow.core.tool_manager import ToolManager

tool_manager = ToolManager(tools=functions)
print(tool_manager)




# execute get_current_weather using function call 

ft.call(**{"location": "San Francisco", "unit": "celsius"})




# async call
import nest_asyncio
from IPython.display import display


nest_asyncio.apply()

# call it synchronously using execute

print(tools[2].execute(**{"a": 10, "b": 2}))

display(await tools[2].acall(**{"a": 10, "b": 2}))
display(await tools[2].execute(**{"a": 10, "b": 2}))





# run sync func

# in sync way

print(tools[1].execute(**{"a": 10, "b": 2}))
print(tools[1].call(**{"a": 10, "b": 2}))

# in async way

display(await tools[1].execute(**{"a": 10, "b": 2}))




# call all the above functions 
import nest_asyncio
import asyncio

nest_asyncio.apply()


import time

async def async_function_1():
    await asyncio.sleep(1)
    return "Function 1 completed"

def sync_function_1():
    time.sleep(1)
    return "Function 1 completed"

async def async_function_2():
    await asyncio.sleep(2)
    return "Function 2 completed"

def sync_function_2():
    time.sleep(2)
    return "Function 2 completed"

async_tool_1 = FunctionTool(async_function_1)
sync_tool_1 = FunctionTool(sync_function_2)
async_tool_2 = FunctionTool(async_function_2)
sync_tool_2 = FunctionTool(sync_function_2)

def run_sync_and_async_mix_without_wait():
    # both sync and async tool can use execute
    # sync tool can also use call
    # takes 5 seconds (1+1+2) + overhead
    start_time = time.time()
    results = [
        async_tool_1.execute(),
        sync_tool_1.call(),
        sync_tool_2.call(),
    ]
    end_time = time.time()
    print(f"run_sync_and_async_mix_without_wait time: {end_time - start_time}")
    return results

async def run_sync_and_async_mix():
    # both sync and async tool can use execute&to_thread
    # async tool can also use acall without to_thread
    # takes a bit over 2 seconds max(2)
    start_time = time.time()
    results = await asyncio.gather(
        async_tool_1.execute(),
        sync_tool_1.execute(),
      
        async_tool_2.acall(),
    )
    end_time = time.time()
    print(f"run_sync_and_async_mix time: {end_time - start_time}")
    return results

# Execute functions
results_without_wait = run_sync_and_async_mix_without_wait()
display(results_without_wait)

results_with_wait = asyncio.run(run_sync_and_async_mix())
display(results_with_wait)




# prepare a template for generator
template = r"""<SYS>You have these tools available:
{% if tools %}
<TOOLS>
{% for tool in tools %}
{{ loop.index }}.
{{tool}}
------------------------
{% endfor %}
</TOOLS>
{% endif %}
<OUTPUT_FORMAT>
{{output_format_str}}
</OUTPUT_FORMAT>
</SYS>
User: {{input_str}}
You:
"""




# let's see how the template can be rendered with tools
from adalflow.core.prompt_builder import Prompt

prompt = Prompt(template=template)
small_tool_manager = ToolManager(tools=tools[:2])

renered_prompt = prompt(tools=tool_manager.yaml_definitions)
print(renered_prompt)




# let's render the output format using Function class 

from adalflow.core.types import Function


output_data_class = Function 
output_format_str = output_data_class.to_json_signature(exclude=["thought"])

renered_prompt= prompt(output_format_str=output_format_str)
print(renered_prompt)




# use FunctionExpression
from adalflow.core.types import FunctionExpression

output_data_class = FunctionExpression
output_format_str = output_data_class.to_json_signature(exclude=["thought"])
print(prompt(output_format_str=output_format_str))




# let's adds more instruction and this time, we will use JsonOutputParser

from adalflow.components.output_parsers import JsonOutputParser

func_parser = JsonOutputParser(data_class=Function)
instructions = func_parser.format_instructions(exclude=["thought"])
print(instructions)




# create the generator
from adalflow.core.generator import Generator
from adalflow.core.types import ModelClientType

model_kwargs = {"model": "gpt-3.5-turbo"}
prompt_kwargs = {
    "tools": tool_manager.yaml_definitions,
    "output_format_str": func_parser.format_instructions(
        exclude=["thought", "args"]
    ),
}
generator = Generator(
    model_client=ModelClientType.OPENAI(),
    model_kwargs=model_kwargs,
    template=template,
    prompt_kwargs=prompt_kwargs,
    output_processors=func_parser,
)
generator




arr = np.array([[1, 2], [3, 4]])
numpy_sum(arr)




# call queries
queries = [
        "add 2 and 3",
        "search for something",
        "add points (1, 2) and (3, 4)",
        "sum numpy array with arr = np.array([[1, 2], [3, 4]])",
        "multiply 2 with local variable x",
        "divide 2 by 3",
        "Add 5 to variable y",
    ]




for idx, query in enumerate(queries):
    prompt_kwargs = {"input_str": query}
    print(f"\n{idx} Query: {query}")
    print(f"{'-'*50}")
    try:
        result = generator(prompt_kwargs=prompt_kwargs)
        # print(f"LLM raw output: {result.raw_response}")
        func = Function.from_dict(result.data)
        print(f"Function: {func}")
        func_output= tool_manager.execute_func(func)
        display(f"Function output: {func_output}")
    except Exception as e:
        print(f"Failed to execute the function for query: {query}, func: {result.data}, error: {e}")


# Problems with Function directly:
# 1. difficult to support data types. Unless to update the function to use dict version of the data types to do it.
# 
# ```python
# def add_points(p1: dict, p2: dict) -> dict:
#     p1 = Point(**p1)
#     p2 = Point(**p2)
#     return add_points_tool.fn(p1, p2).__dict__
# ```
# 2. difficult to use variable as arguments. [TODO: find a proper way to demonstrate it]








# let's use FunctionExpression to call the function instead 

from adalflow.core.types import FunctionExpression

output_data_class = FunctionExpression
output_format_str = output_data_class.to_yaml_signature(exclude=["thought"])
print(output_format_str)

# lets' add one example to be more robust that they should call it with function call expression
example = FunctionExpression.from_function(thought=None, func=add_points, **{"p1": Point(1, 2), "p2": Point(3, 4)})
print(example)




# also use json output parser and create a new generator

parser = JsonOutputParser(data_class=FunctionExpression, example=example)
instructions = parser.format_instructions(exclude=["thought"])

prompt_kwargs = {
        "tools": [tool.definition.to_yaml() for tool in tools],
        "output_format_str": parser.format_instructions(exclude=["thought"]),
    }
generator = Generator(
    model_client=ModelClientType.OPENAI(),
    model_kwargs=model_kwargs,
    template=template,
    prompt_kwargs=prompt_kwargs,
    output_processors=parser
)

generator.print_prompt(**prompt_kwargs)




import ast
import builtins
import contextlib
import ctypes
import sys
import threading
import time

# Define a list of safe built-ins
SAFE_BUILTINS = {
    'abs': abs,
    'all': all,
    'any': any,
    'bin': bin,
    'bool': bool,
    'bytearray': bytearray,
    'bytes': bytes,
    'callable': callable,
    'chr': chr,
    'complex': complex,
    'dict': dict,
    'divmod': divmod,
    'enumerate': enumerate,
    'filter': filter,
    'float': float,
    'format': format,
    'frozenset': frozenset,
    'getattr': getattr,
    'hasattr': hasattr,
    'hash': hash,
    'hex': hex,
    'int': int,
    'isinstance': isinstance,
    'issubclass': issubclass,
    'iter': iter,
    'len': len,
    'list': list,
    'map': map,
    'max': max,
    'min': min,
    'next': next,
    'object': object,
    'oct': oct,
    'ord': ord,
    'pow': pow,
    'range': range,
    'repr': repr,
    'reversed': reversed,
    'round': round,
    'set': set,
    'slice': slice,
    'sorted': sorted,
    'str': str,
    'sum': sum,
    'tuple': tuple,
    'type': type,
    'zip': zip,
}

# Define a context manager to limit execution time
# Create a sandbox execution function
def sandbox_exec(code, context=SAFE_BUILTINS, timeout=5):

    try:
        compiled_code = compile(code, '<string>', 'exec')

        # Result dictionary to store execution results
        result = {
            "output" : None,
            "error" : None
        }

        # Define a target function for the thread
        def target():
            try:
                # Execute the code
                exec(compiled_code, context, result)
            except Exception as e:
                result["error"] = e
            

        # Create a thread to execute the code
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)

        # Check if the thread is still alive (timed out)
        if thread.is_alive():
            result["error"] = TimeoutError("Execution timed out")
            raise TimeoutError("Execution timed out")
    except Exception as e:
        print(f"Errpr at sandbox_exec: {e}")
        raise e

    return result

# Example usage
code = """
def add(a, b+5):
    return a + b

output = add(1, 2+y)
"""

try:
    result = sandbox_exec(code)
    print("Sandbox output:", result)
except TimeoutError as e:
    print(e)
except Exception as e:
    print("Sandbox error:", e)




# run the generator but we will use FunctionTool.parse_function_call_expr and have a context map 

all_functions_dict.update(
    {
    "Point": Point,
    # support numpy
    "np": np,
    "np.ndarray": np.ndarray,
    "array": np.array,
    "arr": arr,
    "np.array": np.array,
    "x": x
    }
)
y=4
print(all_functions_dict)
for query in queries+["Add 5 to variable y"]:

    try:
        print(f"Query: {query}")
        prompt_kwargs = {"input_str": query}
        result = generator(prompt_kwargs=prompt_kwargs)
        print(result)

        func_expr = FunctionExpression.from_dict(result.data)

        print(func_expr)
        assert isinstance(func_expr, FunctionExpression), f"Expected FunctionExpression, got {type(result.data)}"

        # more secure way to handle function call
        func: Function = FunctionTool.parse_function_call_expr(expr=func_expr.action, context_map=all_functions_dict)
        print(func)
        fun_output = all_functions_dict[func.name](*func.args, **func.kwargs)
        print("func output:", fun_output)

        print(f"func expr: {func_expr.action}")

        # eval without security check by using eval directly
        # less secure but even more powerful and flexible
        fun_output = eval(func_expr.action)
        print("func output:", fun_output)

        # sandbox_exec
        action = "output=" + func_expr.action
        result = sandbox_exec(action, context={**SAFE_BUILTINS, **all_functions_dict})
        print("sandbox output:", result)
    except Exception as e:
        print(e)
        print(f"Failed to execute the function for query: {query}, func: {result.data}, error: {e}")
        try:
            fun_output = eval(func_expr.action)
            print("func output:", fun_output)

            #sandbox_exec
            action = "output=" + func_expr.action
            result = sandbox_exec(action, context={**SAFE_BUILTINS, **all_functions_dict})
            print("sandbox output:", result)
        except Exception as e:
            print(e)
            print(f"Failed to execute the function for query: {query}, func: {result.data}, error: {e}")


# Multiple function calls



multple_function_call_template = r"""<SYS>You have these tools available:
{% if tools %}
<TOOLS>
{% for tool in tools %}
{{ loop.index }}.
{{tool}}
------------------------
{% endfor %}
</TOOLS>
{% endif %}
<OUTPUT_FORMAT>
Here is how you call one function.
{{output_format_str}}
Return a List using `[]` of the above JSON objects. You can have length of 1 or more.
Do not call multiple functions in one action field.
</OUTPUT_FORMAT>
<SYS>
{{input_str}}
You:
"""




queries = ["add 2 and 3", "search for something", "add points (1, 2) and (3, 4)", "sum numpy array with arr = np.array([[1, 2], [3, 4]])", "multiply 2 with local variable x", "divide 2 by 3"]

from adalflow.components.output_parsers import ListOutputParser
from adalflow.core.string_parser import JsonParser # improve a list of json

preset_prompt_kwargs = {
        "tools": [tool.definition.to_yaml() for tool in tools],
        "output_format_str": parser.format_instructions(exclude=["thought"])
    }
multi_call_gen = Generator(
    model_client=ModelClientType.OPENAI(),
    model_kwargs=model_kwargs,
    template=multple_function_call_template,
    prompt_kwargs=preset_prompt_kwargs,
    output_processors=JsonParser()
)
print(multi_call_gen)
multi_call_gen.print_prompt()




def execute_function_by_parsing(func_expr: FunctionExpression, all_functions_dict: Dict[str, Any]) -> Any:
    func: Function = FunctionTool.parse_function_call_expr(expr=func_expr.action, context_map=all_functions_dict)
    print(func)
    fun_output = all_functions_dict[func.name](*func.args, **func.kwargs)
    print("func output:", fun_output)
    return fun_output


def execute_function_by_eval(func_expr: FunctionExpression) -> Any:

    print(f"func expr: {func_expr.action}")

    # eval without security check by using eval directly
    # less secure but even more powerful and flexible
    fun_output = eval(func_expr.action)
    print("func output:", fun_output)
    return fun_output

def execute_function_by_sandbox(func_expr: FunctionExpression, all_functions_dict: Dict[str, Any]) -> Any:
    # sandbox_exec
    action = "output=" + func_expr.action
    result = sandbox_exec(action, context={**SAFE_BUILTINS, **all_functions_dict})
    print("sandbox output:", result)

    return result




for i in range(0, len(queries), 2):
    query = " and ".join(queries[i:i+2])
    print(f"Query: {query}\n_________________________\n")
    prompt_kwargs = {"input_str": query}
    result = multi_call_gen(prompt_kwargs=prompt_kwargs)
    print(result)

    try:

        func_exprs = [FunctionExpression.from_dict(item) for item in result.data]

        print(func_exprs)
    except Exception as e:
        print(e)
        print(f"Failed to parse the function for query: {query}, func: {result.data}, error: {e}")
        continue
    try:
        func_outputs_1 = [execute_function_by_parsing(func_expr, all_functions_dict) for func_expr in func_exprs]
        print(f"fun_output by parsing: {func_outputs_1}\n_________________________\n")
    except Exception as e:
        print(e)
        print(f"Failed to execute the function for query: {query}, func: {result.data}, error: {e}")

    try:

        func_outputs_2 = [execute_function_by_eval(func_expr) for func_expr in func_exprs]
        print(f"fun_output by eval: {func_outputs_2}\n_________________________\n")
    except Exception as e:
        print(e)
        print(f"Failed to execute the function for query: {query}, func: {result.data}, error: {e}")

    try:

        func_outputs_3 = [execute_function_by_sandbox(func_expr, all_functions_dict) for func_expr in func_exprs]
        print(f"fun_output by sandbox: {func_outputs_3}\n_________________________\n")
    except Exception as e:
        print(e)
        print(f"Failed to execute the function for query: {query}, func: {result.data}, error: {e}")

        




# first check the openai's function call apis

from openai import OpenAI
from openai.types import FunctionDefinition
from adalflow.utils import setup_env
import json

client = OpenAI()

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

def run_conversation():
    # Step 1: send the conversation and available functions to the model
    messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris in celsius?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    print(f"response: {response}")
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    print(f"tool_calls: {tool_calls}")
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_current_weather": get_current_weather,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)# use json.loads to convert a string to a dictionary
            # function_response = function_to_call(
            #     location=function_args.get("location"),
            #     unit=function_args.get("unit"),
            # ) 
            # you have to exactly know the arguments, this does not make sense. How would i know its arguments. **function_args (makes more sense)
            function_response = function_to_call(**function_args)
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        second_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )  # get a new response from the model where it can see the function response
        return second_response
print(run_conversation())









# Function(arguments='{"location": "Tokyo, Japan", "unit": "celsius"}', name='get_current_weather'


# There are two important pieces. Getting function schema is not difficult and can be standarized.
# 
# The second piece is how to call the function, and how to execute it. The how to call the function depends on how we execute it.
# 
# How to execute a function:
# 1. Eval (LLM will output the code to call the function (in string format))-> Language generation.
# 2. We manage a function map, and we ask LLm to output either the code string or a structure with the function name and the arguments. We can use the function map to call the function. If its code string, we will have to parse the function call into the name and the arguments. If its a structure, we will have to convert it to data structure that can be used to call the function.
# 
# There are just so many different ways to do the actual function call, and different LLM might react differetntly in accuracy to each output format.






# Function(arguments='{"location": "Paris, France"}', name='get_current_weather'), type='function')



def get_current_weather(location: str, unit: str = "fahrenheit"):
        """Get the current weather in a given location"""
        if "tokyo" in location.lower():
            return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
        elif "san francisco" in location.lower():
            return json.dumps(
                {"location": "San Francisco", "temperature": "72", "unit": unit}
            )
        elif "paris" in location.lower():
            return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
        else:
            return json.dumps({"location": location, "temperature": "unknown"})





# v2

from adalflow.core.base_data_class import DataClass
from dataclasses import dataclass, field

@dataclass
class Weather(DataClass):
    location: str = field(metadata={"description": "The city and state, e.g. San Francisco, CA"})
    unit: str = field(metadata={"enum": ["celsius", "fahrenheit"]})

def get_current_weather_2(weather: Weather):
    """Get the current weather in a given location"""
    if "tokyo" in weather.location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": weather.unit})
    elif "san francisco" in weather.location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": weather.unit}
        )
    elif "paris" in weather.location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": weather.unit})
    else:
        return json.dumps({"location": weather.location, "temperature": "unknown"})




# Create a tool from the class

tool_2 = FunctionTool.from_defaults(fn=get_current_weather_2)

print(tool_2.metadata.to_json())



# Llamaindex
# 
# 



adalflow_fn_schema =
{
        "type": "object",
        "properties": {
            "weather": {
                "type": "Weather",
                "desc": "The city and state, e.g. San Francisco, CA",
                "enum": [
                    "celsius",
                    "fahrenheit"
                ]
            }
        },
        "required": [
            "weather"
        ],
        "definitions": {
            "weather": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "str"
                    },
                    "unit": {
                        "type": "str"
                    }
                },
                "required": [
                    "location",
                    "unit"
                ]
            }
        }
}




llama_fn_schema = {
    "type": "object",
    "properties": {"weather": {"$ref": "#/definitions/Weather"}},
    "required": ["weather"],
    "definitions": {
        "Weather": {
            "title": "Weather",
            "type": "object",
            "properties": {
                "location": {
                    "title": "Location",
                    "desc": "The city and state, e.g. San Francisco, CA",
                    "type": "string",
                },
                "unit": {
                    "title": "Unit",
                    "enum": ["celsius", "fahrenheit"],
                    "type": "string",
                },
            },
            "required": ["location", "unit"],
            "additionalProperties": false,
        }
    },
}




# level 1, call function with default python data types
# such as str, int, float, list, dict, etc.

def _get_current_weather(location: str, unit: str = "fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": unit}
        )
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})




# prepare function tool 
weather_tool = FunctionTool.from_defaults(fn=_get_current_weather)
print(weather_tool)




# prepare a minimal function calling template 
template = r"""<SYS>You have these tools available:
    <TOOLS>
    {% for tool in tools %}
    {{ loop.index }}. ToolName: {{ tool.metadata.name }}
        Tool Description: {{ tool.metadata.description }}
        Tool Parameters: {{ tool.metadata.fn_schema_str }}   
    __________
    {% endfor %}
    </TOOLS>
    {{output_format_str}}
    </SYS>
    User: {{input_str}}
    You:
    """

multiple_function_call_template = r"""<SYS>You can answer user query with these tools:
    <TOOLS>
    {% for tool in tools %}
    {{ loop.index }}. ToolName: {{ tool.metadata.name }}
        Tool Description: {{ tool.metadata.description }}
        Tool Parameters: {{ tool.metadata.fn_schema_str }}   
    __________
    {% endfor %}
    </TOOLS>
    You can call multiple tools by return a list of the following format:
    {{output_format_str}}
    </SYS>
    User: {{input_str}}
    You:
    """

from typing import Dict, Any
from adalflow.core.generator import Generator
from adalflow.core.types import ModelClientType
from adalflow.components.output_parsers import YamlOutputParser

model_kwargs = {"model": "gpt-3.5-turbo", "temperature": 0.3, "stream": False}

@dataclass
class Function(DataClass):
    name: str = field(metadata={"desc": "The name of the function"})
    args: Dict[str, Any] = field(metadata={"desc": "The arguments of the function"})

generator = Generator(
    model_client=ModelClientType.OPENAI(),
    model_kwargs=model_kwargs,
    template=template,
    prompt_kwargs={
        # "tools": [weather_tool],
        "output_format_str": YamlOutputParser(Function).format_instructions(),
        # "output_format_str": Function.to_yaml_signature(),
    },
    output_processors=YamlOutputParser(Function),
)
generator




# check the prompt

input_str = "What's the weather like in San Francisco, Tokyo, and Paris in celsius?"

generator.print_prompt(input_str=input_str, tools=[weather_tool])




prompt_kwargs = {
    "input_str": input_str,
    "tools": [weather_tool],
}
output = generator(prompt_kwargs=prompt_kwargs)
structured_output = Function.from_dict(output.data)
print(structured_output)




# call the function

function_map = {
    "_get_current_weather": weather_tool
}

function_name = structured_output.name
function_args = structured_output.args
function_to_call = function_map[function_name]
function_response = function_to_call(**function_args)
print(function_response)


# # multiple function calls



generator = Generator(
    model_client=ModelClientType.OPENAI(),
    model_kwargs=model_kwargs,
    template=multiple_function_call_template,
    prompt_kwargs={
        # "tools": [weather_tool],
        "output_format_str": YamlOutputParser(Function).format_instructions(),
        # "output_format_str": Function.to_yaml_signature(),
    },
    output_processors=YamlOutputParser(Function),
)
generator




# run the query

output = generator(prompt_kwargs=prompt_kwargs)
list_structured_output = [Function.from_dict(item) for item in output.data]
print(output)
print(list_structured_output)




for structured_output in list_structured_output:
    function_name = structured_output.name
    function_args = structured_output.args
    function_to_call = function_map[function_name]
    function_response = function_to_call(**function_args)
    print(function_response)


# 



from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class Address:
    street: str
    city: str
    zipcode: str

@dataclass
class Person:
    name: str
    age: int
    address: Address

# Example instance of the nested dataclasses
person = Person(name="John Doe", age=30, address=Address(street="123 Main St", city="Anytown", zipcode="12345"))
print(person)

def to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "__dataclass_fields__"):
        return {key: to_dict(value) for key, value in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: to_dict(value) for key, value in obj.items()}
    else:
        return obj

# Convert the person instance to a dictionary
person_dict = to_dict(person)
print(person_dict)




from typing import List
@dataclass
class Address:
    street: str
    city: str
    zipcode: str

@dataclass
class Person:
    name: str
    age: int
    addresses: List[Address]

# Example instance of the nested dataclasses
person = Person(name="John Doe", age=30, addresses=[Address(street="123 Main St", city="Anytown", zipcode="12345"), Address(street="456 Elm St", city="Othertown", zipcode="67890")])
print(person)




# Convert the person instance to a dictionary
person_dict = to_dict(person)
print(person_dict)




from typing import List, Dict, Optional
def dataclass_obj_to_dict(
    obj: Any, exclude: Optional[Dict[str, List[str]]] = None, parent_key: str = ""
) -> Dict[str, Any]:
    r"""Convert a dataclass object to a dictionary.

    Supports nested dataclasses, lists, and dictionaries.
    Allow exclude keys for each dataclass object.
    Example:

    .. code-block:: python

       from dataclasses import dataclass
       from typing import List

       @dataclass
       class TrecData:
           question: str
           label: int

       @dataclass
       class TrecDataList:

           data: List[TrecData]
           name: str

       trec_data = TrecData(question="What is the capital of France?", label=0)
       trec_data_list = TrecDataList(data=[trec_data], name="trec_data_list")

       dataclass_obj_to_dict(trec_data_list, exclude={"TrecData": ["label"], "TrecDataList": ["name"]})

       # Output:
       # {'data': [{'question': 'What is the capital of France?'}], 'name': 'trec_data_list'}

    """
    if exclude is None:
        exclude = {}

    obj_class_name = obj.__class__.__name__
    current_exclude = exclude.get(obj_class_name, [])

    if hasattr(obj, "__dataclass_fields__"):
        return {
            key: dataclass_obj_to_dict(value, exclude, parent_key=key)
            for key, value in obj.__dict__.items()
            if key not in current_exclude
        }
    elif isinstance(obj, list):
        return [dataclass_obj_to_dict(item, exclude, parent_key) for item in obj]
    elif isinstance(obj, dict):
        return {
            key: dataclass_obj_to_dict(value, exclude, parent_key)
            for key, value in obj.items()
        }
    else:
        return obj

from dataclasses import dataclass
from typing import List

@dataclass
class TrecData:
    question: str
    label: int

@dataclass
class TrecDataList:

    data: List[TrecData]
    name: str

trec_data = TrecData(question="What is the capital of France?", label=0)
trec_data_list = TrecDataList(data=[trec_data], name="trec_data_list")

dataclass_obj_to_dict(trec_data_list, exclude={"TrecData": ["label"], "TrecDataList": ["name"]})




from typing import Type
def dataclass_obj_from_dict(cls: Type[Any], data: Dict[str, Any]) -> Any:
    if hasattr(cls, "__dataclass_fields__"):
        fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
        return cls(**{key: dataclass_obj_from_dict(fieldtypes[key], value) for key, value in data.items()})
    elif isinstance(data, list):
        return [dataclass_obj_from_dict(cls.__args__[0], item) for item in data]
    elif isinstance(data, dict):
        return {key: dataclass_obj_from_dict(cls.__args__[1], value) for key, value in data.items()}
    else:
        return data




dataclass_obj_from_dict(TrecDataList, dataclass_obj_to_dict(trec_data_list))




dataclass_obj_from_dict(TrecDataList, dataclass_obj_to_dict(trec_data_list, exclude={"TrecData": ["label"], "TrecDataList": ["name"]}))

