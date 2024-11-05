#!/usr/bin/env python
# coding: utf-8

# We have already seen how the RAG is implemented in data.
# In this note, we will focus more on how to make each component more configurable, 
# espeically the data processing pipeline to help us with experiments where we will see how useful they are in benchmarking.



# the data pipeline and the backend data processing
from adalflow.core.embedder import Embedder 
from adalflow.core.types import ModelClientType
from adalflow.components.data_process import TextSplitter, ToEmbeddings
from adalflow.core.container import Sequential

def prepare_data_pipeline():
    model_kwargs = {
        "model": "text-embedding-3-small",
        "dimensions": 256,
        "encoding_format": "float",
    }

    splitter_config = {
        "split_by": "word",
        "split_length": 50,
        "split_overlap": 10
    }

    splitter = TextSplitter(**splitter_config)
    embedder = Embedder(model_client =ModelClientType.OPENAI(), model_kwargs=model_kwargs)
    embedder_transformer = ToEmbeddings(embedder, batch_size=2)
    data_transformer = Sequential(splitter, embedder_transformer)
    print(data_transformer)

