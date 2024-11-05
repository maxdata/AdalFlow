#!/usr/bin/env python
# coding: utf-8



from dataclasses import dataclass, field

@dataclass
class Question:
    question: str = field(
        metadata={"desc": "The question asked by the user"}
    )
    metadata: dict = field(
        metadata={"desc": "The metadata of the question"}, default_factory=dict
    )

@dataclass
class TrecData:
    question: Question = field(
        metadata={"desc": "The question asked by the user"}
    ) # Required field, you have to provide the question field at the instantiation
    label: int = field(
        metadata={"desc": "The label of the question"}, default=0
    ) # Optional field




# dataclass itself is powerful, but it can not reconstruct nested dataclass
example = TrecData(Question("What is the capital of France?"), 1)
print(example)

from dataclasses import asdict
print(asdict(example))
reconstructed = TrecData(**asdict(example))
print(reconstructed)
print(reconstructed == example)




# it does not allow required field after optional field
@dataclass
class TrecData2:
    question: Question = field(
        metadata={"desc": "The question asked by the user"}
    ) # Required field, you have to provide the question field at the instantiation
    label: int = field(
        metadata={"desc": "The label of the question"}, default=0
    ) # Optional field
    metadata: dict = field(
        metadata={"desc": "The metadata of the question"}
    ) # required field




# lets see what DataClass can do 
# 1. allow required field after optional field using required_field on default_factory

from adalflow.core import DataClass, required_field

@dataclass
class TrecData2(DataClass):
    question: Question = field(
        metadata={"desc": "The question asked by the user"}
    ) # Required field, you have to provide the question field at the instantiation
    label: int = field(
        metadata={"desc": "The label of the question"}, default=0
    ) # Optional field
    metadata: dict = field(
        metadata={"desc": "The metadata of the question"}, default_factory=required_field()
    ) # required field




# 2. allow you to reconstructed nested dataclass
# You just have to make sure the class you are handling are subclass of DataClass, the child class can be native dataclass

example = TrecData2(Question("What is the capital of France?"), 1, {"key": "value"})
print(example)

dict_example = TrecData2.to_dict(example) # use as if its a class method
print(dict_example)

dict_example_2 = example.to_dict() # use it  as instance method
print(dict_example)

reconstructed = TrecData2.from_dict(dict_example)
print(reconstructed)

print(reconstructed == example)
print(dict_example == dict_example_2)




# Lets exclude fields too

# Note: better not to exclude required fields, as it will run issues using from_dict
# you can use it if you dont mind to reconstruct

# exclude field of only the parent class
dict_exclude = example.to_dict(exclude=["metadata"])
print(dict_exclude)

# exclude field of the parent and child class
dict_exclude = example.to_dict(exclude={"TrecData2": ["metadata"], "Question": ["metadata"]})
print(dict_exclude)




# lets do the yaml and json string for demonstraing the data example

json_str = example.to_json()
print(json_str)

yaml_str = example.to_yaml()
print(yaml_str)

reconstructed_from_json = TrecData2.from_json(json_str)
print(reconstructed_from_json)
print(reconstructed_from_json == example)

reconstructed_from_yaml = TrecData2.from_yaml(yaml_str)
print(reconstructed_from_yaml)
print(reconstructed_from_yaml == example)




# use with DataClassFormatType and format_example_str

from adalflow.core import DataClassFormatType

example_str = example.format_example_str(DataClassFormatType.EXAMPLE_JSON)
print(example_str)

example_str = example.format_example_str(DataClassFormatType.EXAMPLE_YAML)
print(example_str)





# Now, lets check the data format using class method without instance
# schema, you can choose to only use properties 

schema = TrecData2.to_schema()
schema




# schema with exclude
schema_exclude = TrecData2.to_schema(exclude={"TrecData2": ["metadata"], "Question": ["metadata"]})
schema_exclude




# signature, json_signature

json_signature = TrecData2.to_json_signature()
print(json_signature)




# exclude field of the parent and child class

json_signature_exclude = TrecData2.to_json_signature(exclude={"TrecData2": ["metadata"], "Question": ["metadata"]})
print(json_signature_exclude)




# only exclude the parent class

json_signature_exclude = TrecData2.to_json_signature(exclude=["metadata"])
print(json_signature_exclude)




# signature, yaml_signature

yaml_signature = TrecData2.to_yaml_signature()
print(yaml_signature)




# use the DataClassFormatType to control it 

from adalflow.core import DataClassFormatType

json_signature = TrecData2.format_class_str(DataClassFormatType.SIGNATURE_JSON)
print(json_signature)

yaml_signature = TrecData2.format_class_str(DataClassFormatType.SIGNATURE_YAML)
print(yaml_signature)

schema = TrecData2.format_class_str(DataClassFormatType.SCHEMA)
print(schema)




# load with customizd from dict
from typing import Dict
@dataclass
class OutputFormat(DataClass):
    thought: str = field(
        metadata={
            "desc": "Your reasoning to classify the question to class_name",
        }
    )
    class_name: str = field(metadata={"desc": "class_name"})
    class_index: int = field(metadata={"desc": "class_index in range[0, 5]"})

    @classmethod
    def from_dict(cls, data: Dict[str, object]):
        _COARSE_LABELS_DESC = [
            "Abbreviation",
            "Entity",
            "Description and abstract concept",
            "Human being",
            "Location",
            "Numeric value",
        ]
        data = {
            "thought": None,
            "class_index": data["coarse_label"],
            "class_name": _COARSE_LABELS_DESC[data["coarse_label"]],
        }
        return super().from_dict(data)

data = OutputFormat.from_dict({"coarse_label": 1})
print(data)

