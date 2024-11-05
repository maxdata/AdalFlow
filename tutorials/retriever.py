#!/usr/bin/env python
# coding: utf-8

# We start with high-precision and high-recall retrieval methods as AdalFlow helps you optimize the later stage of your search/retrieval pipeline. As the first stage is often comes with cloud db providers with their search and filter support.

# ## FAISS retriever (simple)
# 
# We mainly use this to quickly show how to implement semantic search as retriever.



# decide a meaningful query and a list of documents
query_1 = "What are the benefits of renewable energy?" # gt is [0, 3]
query_2 = "How do solar panels impact the environment?" # gt is [1, 2]

documents =[
    {
        "title": "The Impact of Renewable Energy on the Economy",
        "content": "Renewable energy technologies not only help in reducing greenhouse gas emissions but also contribute significantly to the economy by creating jobs in the manufacturing and installation sectors. The growth in renewable energy usage boosts local economies through increased investment in technology and infrastructure."
    },
    {
        "title": "Understanding Solar Panels",
        "content": "Solar panels convert sunlight into electricity by allowing photons, or light particles, to knock electrons free from atoms, generating a flow of electricity. Solar panels are a type of renewable energy technology that has been found to have a significant positive effect on the environment by reducing the reliance on fossil fuels."
    },
    {
        "title": "Pros and Cons of Solar Energy",
        "content": "While solar energy offers substantial environmental benefits, such as reducing carbon footprints and pollution, it also has downsides. The production of solar panels can lead to hazardous waste, and large solar farms require significant land, which can disrupt local ecosystems."
    },
    {
        "title":  "Renewable Energy and Its Effects",
        "content": "Renewable energy sources like wind, solar, and hydro power play a crucial role in combating climate change. They do not produce greenhouse gases during operation, making them essential for sustainable development. However, the initial setup and material sourcing for these technologies can still have environmental impacts."
    }
]




# create an embedder
from adalflow.core.embedder import Embedder 
from adalflow.core.types import ModelClientType


model_kwargs = {
    "model": "text-embedding-3-small",
    "dimensions": 256,
    "encoding_format": "float",
}

embedder = Embedder(model_client =ModelClientType.OPENAI(), model_kwargs=model_kwargs)
embedder




# the documents can fit into a batch, thus we only need the simple embedder
# embedder takes a list of string. we will pass only the content of the documents
output = embedder(input=[doc["content"] for doc in documents])
print(output.embedding_dim, output.length, output)




# prepare the retriever

from adalflow.components.retriever import FAISSRetriever

# pass the documents in the initialization 
documents_embeddings = [x.embedding for x in output.data]
retriever = FAISSRetriever(top_k=2, embedder=embedder, documents=documents_embeddings)
retriever




# execute the retriever
output_1 = retriever(input=query_1)
output_2 = retriever(input=query_2)
output_3 = retriever(input = [query_1, query_2])
print(output_1)
print(output_2)
print(output_3)




# second, we dont pass documents in init, and instead pass it with method build_index_from_documents

retriever_1 = FAISSRetriever(top_k=2, embedder=embedder)
print(retriever_1)
retriever_1.build_index_from_documents(documents=documents_embeddings)
print(retriever_1)

output_1 = retriever_1(input=query_1)
output_2 = retriever_1(input=query_2)
output_3 = retriever_1(input = [query_1, query_2])
print(output_1)
print(output_2)
print(output_3)


# ## BM25 retriever (simple)



from adalflow.components.retriever import BM25Retriever

document_map_func = lambda x: x["content"]

bm25_retriever = BM25Retriever(top_k=2, documents=documents, document_map_func=document_map_func)
print(bm25_retriever)




# show how a word splitter and a token splitter differs

from adalflow.components.retriever.bm25_retriever import split_text_by_word_fn_then_lower_tokenized, split_text_by_word_fn

query_1_words = split_text_by_word_fn(query_1)
query_1_tokens = split_text_by_word_fn_then_lower_tokenized(query_1)

print(query_1_words)
print(query_1_tokens)




output_1 = bm25_retriever(input=query_1)
output_2 = bm25_retriever(input=query_2)
output_3 = bm25_retriever(input = [query_1, query_2])
print(output_1)
print(output_2)
print(output_3)




states = bm25_retriever.to_dict()
print(states)




# use short queries, it performs slightly better

query_1_short = "renewable energy?"  # gt is [0, 3]
query_2_short = "solar panels?"  # gt is [1, 2]

output_1 = bm25_retriever(input=query_1_short)
output_2 = bm25_retriever(input=query_2_short)
output_3 = bm25_retriever(input = [query_1_short, query_2_short])
print(output_1)
print(output_2)
print(output_3)




# use both title and content
document_map_func = lambda x: x["title"] + " " + x["content"]

print(documents)
bm25_retriever.build_index_from_documents(documents=documents, document_map_func=document_map_func)

output_1 = bm25_retriever(input=query_1_short)
output_2 = bm25_retriever(input=query_2_short)
output_3 = bm25_retriever(input = [query_1_short, query_2_short])
print(output_1)
print(output_2)
print(output_3)


# ## Reranker (simple)



# !poetry add cohere --group dev




from adalflow.components.retriever import RerankerRetriever

model_client = ModelClientType.COHERE()
model_kwargs = {"model": "rerank-english-v3.0"}


reranker = RerankerRetriever(
    top_k=2, model_client=model_client, model_kwargs=model_kwargs
)
print(reranker)




# build index and run queries
document_map_func = lambda x: x["content"]
reranker.build_index_from_documents(documents=documents, document_map_func=document_map_func)

print(reranker)




# run queries
output_1 = reranker(input=query_1)
output_2 = reranker(input=query_2)
output_3 = reranker(input = [query_1, query_2])
print(output_1)
print(output_2)
print(output_3)




# use transformer client

model_client = ModelClientType.TRANSFORMERS()
model_kwargs = {"model": "BAAI/bge-reranker-base"}

reranker = RerankerRetriever(
    top_k=2,
    model_client=model_client,
    model_kwargs=model_kwargs,
    documents=documents,
    document_map_func=document_map_func,
)
print(reranker)




# run queries
import torch
# Set the number of threads for PyTorch, avoid segementation fault
torch.set_num_threads(1)
torch.set_num_interop_threads(1)




output_1 = reranker(input=query_1)
output_2 = reranker(input=query_2)
output_3 = reranker(input = [query_1, query_2])
print(output_1)
print(output_2)
print(output_3)


# As we see the second query is missing one. But Semantically, these documents might be close.
# If we use top_k = 3, the genearator might be able to filter out the irrelevant one and eventually give out the right final response.



# try to use title this time
document_map_func = lambda x: x["title"] + " " + x["content"]

reranker.build_index_from_documents(documents=documents, document_map_func=document_map_func)

# run queries
output_1 = reranker(input=query_1)
output_2 = reranker(input=query_2)
output_3 = reranker(input = [query_1, query_2])
print(output_1)
print(output_2)
print(output_3)


# ## LLM as retriever
# 
# (1) Directly return the doc_indices from the LLM model.



from adalflow.components.retriever import LLMRetriever

model_client = ModelClientType.OPENAI()
model_kwargs = {
    "model": "gpt-4o",
}
document_map_func = lambda x: x["content"]
llm_retriever = LLMRetriever(
        top_k=2, 
        model_client=model_client, 
        model_kwargs=model_kwargs, 
        documents=documents, 
        document_map_func=document_map_func
    )
print(llm_retriever)




# run queries
output_1 = llm_retriever(input=query_1)
output_2 = llm_retriever(input=query_2)
output_3 = llm_retriever(input = [query_1, query_2])
print(output_1)
print(output_2)
print(output_3)




# you should try both gpt-3.5-turbo and gpt-4o
# you can use a different model without reinitializing the retriever
model_kwargs = {
    "model": "gpt-3.5-turbo",
}
output_1 = llm_retriever(model_kwargs=model_kwargs, input=query_1)
output_2 = llm_retriever(model_kwargs=model_kwargs, input=query_2)
output_3 = llm_retriever(model_kwargs=model_kwargs, input = [query_1, query_2])
print(output_1)
print(output_2)
print(output_3)


# ## LLMRetriever
# 
# The indexing process is to form prompt using the targeting documents and set up the top_k parameter.
# The ``retrieve`` is to run the ``generator`` and parse the response to standard ``RetrieverOutputType`` which is a list of 
# ``RetrieverOutput``. Each ``RetrieverOutput`` contains the document id and the score.



# prepare the document
get_ipython().system("mkdir -p 'data/paul_graham/'")
get_ipython().system("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")




# use fsspec to read the document
get_ipython().system('pip install fsspec')




import fsspec
import os
import time
def get_local_file_metadata(file_path: str):
    stat = os.stat(file_path)
    return {
        'size': stat.st_size,  # File size in bytes
        'creation_date': time.ctime(stat.st_ctime),  # Creation time
        'last_modified_date': time.ctime(stat.st_mtime)  # Last modification time
    }


def load_text_file(file_path: str) -> str:
    """
    Loads a text file from the specified path using fsspec.

    Args:
        file_path (str): The path to the text file. This can be a local path or a URL for a supported file system.

    # Example usage with a local file
    local_file_path = 'file:///path/to/localfile.txt'
    print(load_text_file(local_file_path))

    # Example usage with an S3 file
    s3_file_path = 's3://mybucket/myfile.txt'
    print(load_text_file(s3_file_path))

    # Example usage with a GCS file
    gcs_file_path = 'gcs://mybucket/myfile.txt'
    print(load_text_file(gcs_file_path))

    # Example usage with an HTTP file
    http_file_path = 'https://example.com/myfile.txt'
    print(load_text_file(http_file_path))

    Returns:
        str: The content of the text file.
    """
    with fsspec.open(file_path, 'r') as file:
        content = file.read()
    return content




text = load_text_file('paul_graham/paul_graham_essay.txt')
file_metadata = get_local_file_metadata('paul_graham/paul_graham_essay.txt')
print(text[:1000])
print(file_metadata)




# split the documents

from adalflow.components.data_process import DocumentSplitter
from adalflow.core.types import Document

# sentence splitting is confusing, the length needs to be smaller
metadata = {"title": "Paul Graham's essay", "path": "data/paul_graham/paul_graham_essay.txt"}
metadata.update(file_metadata)
documents = [Document(text = text, meta_data = metadata)]
splitter = DocumentSplitter(split_by="word", split_length=800, split_overlap=200)

print(documents)
print(splitter)




token_limit = 16385

# compute the maximum number of splitted_documents with split length = 800 and overlap = 200
# total of 28 subdocuments now

16385 // 800


# From the document structure, we can see the ``estimated_num_tokens=16534`` this will help us
# adapt our retriever.



# split the document
splitted_documents = splitter(documents = documents)
print(splitted_documents[0], len(splitted_documents))




from adalflow.components.retriever import LLMRetriever
from adalflow.components.model_client import OpenAIClient

from adalflow.tracing import trace_generator_call

from adalflow.utils import setup_env

# 1. set up the tracing for failed call as the retriever has generator attribute

@trace_generator_call(save_dir="tutorials/traces")
class LoggedLLMRetriever(LLMRetriever):
    pass
top_k = 2
retriever = LoggedLLMRetriever(
    top_k = top_k, model_client=OpenAIClient(), model_kwargs={"model": "gpt-3.5-turbo"}
)

retriever.build_index_from_documents(documents=[doc.text for doc in splitted_documents[0:16]])

print(retriever)
retriever.generator.print_prompt()


# Note: We need to know the ground truth, you can save the splitted documents and then label the data.
# 
# Here we did that, the ground truth is (indices)



query = "What happened at Viaweb and Interleaf?"
output = retriever(input=query)
print(output)




# output[0].documents = [splitted_documents[idx] for idx in output[0].doc_indices]
for per_query_output in output:
    per_query_output.documents = [splitted_documents[idx] for idx in per_query_output.doc_indices]
print("output.documents", output[0].documents)
len(output)




# check the first document
print(output[0].documents[0].text)
print("interleaf" in output[0].documents[0].text.lower())
print("viaweb" in output[0].documents[0].text.lower())




# check the second document
print(output[0].documents[1].text)
print("interleaf" in output[0].documents[1].text.lower())
print("viaweb" in output[0].documents[1].text.lower())


# ## Reranker
# 



# from adalflow.components.retriever import RerankerRetriever

# query = "Li"
# strings = ["Li", "text2"]

# retriever = RerankerRetriever(top_k=1)
# print(retriever)
# retriever.build_index_from_documents(documents=documents)
# print(retriever.documents)
# output = retriever.retrieve(query)
# print(output)




# retriever.build_index_from_documents(documents=strings)




# output = retriever.retrieve(query)


# ## FAISSRetriever
# 
# To use Semantic search, we very likely need TextSplitter and compute the embeddings. This data-preprocessing is more use-case specific and should be better to be done by users in data transformation stage. Then we can treat these embeddings as the input documents.
# 
# In this case, the real index is the splitted documents along with its embeddings. We will use ``LocalDB`` to handle the data transformation and the storage of the index.
# 
# 



from adalflow.core.db import LocalDB

db = LocalDB()
db.load_documents(documents)
len(db.documents)


# Let us see how to create data transformers using only the component config



# create data transformer
data_transformer_config = {  # attribute and its config to recreate the component
        "embedder":{
            "component_name": "Embedder",
            "component_config": {
                "model_client": {
                    "component_name": "OpenAIClient",
                    "component_config": {},
                },
                "model_kwargs": {
                    "model": "text-embedding-3-small",
                    "dimensions": 256,
                    "encoding_format": "float",
                },
            },
        },
        "document_splitter": {
            "component_name": "DocumentSplitter",
            "component_config": {
                "split_by": "word",
                "split_length": 400,
                "split_overlap": 200,
            },
        },
        "to_embeddings": {
            "component_name": "ToEmbeddings",
            "component_config": {
                "vectorizer": {
                    "component_name": "Embedder",
                    "component_config": {
                        "model_client": {
                            "component_name": "OpenAIClient",
                            "component_config": {},
                        },
                        "model_kwargs": {
                            "model": "text-embedding-3-small",
                            "dimensions": 256,
                            "encoding_format": "float",
                        },
                    },
                    # the other config is to instantiate the entity (class and function) with the given config as arguments
                    # "entity_state": "storage/embedder.pkl", # this will load back the state of the entity
                },
                "batch_size": 100,
            },
        },
    }




from adalflow.utils.config import new_components_from_config

components = new_components_from_config(data_transformer_config)
print(components)




from adalflow.core.component import Sequential

data_transformer = Sequential(components["document_splitter"], components["to_embeddings"])
data_transformer


# The above code is equivalent to the code with config
# 
# ```python
# 
#         vectorizer = Embedder(
#             model_client=OpenAIClient(),
#             # batch_size=self.vectorizer_settings["batch_size"],
#             
#             model_kwargs=self.vectorizer_settings["model_kwargs"],
#         )
#         # TODO: check document splitter, how to process the parent and order of the chunks
#         text_splitter = DocumentSplitter(
#             split_by=self.text_splitter_settings["split_by"],
#             split_length=self.text_splitter_settings["chunk_size"],
#             split_overlap=self.text_splitter_settings["chunk_overlap"],
#         )
#         self.data_transformer = Sequential(
#             text_splitter,
#             ToEmbeddings(
#                 vectorizer=vectorizer,
#                 batch_size=self.vectorizer_settings["batch_size"],
#             ),
#         )
# ```
# 
# Config:
# 
# ```yaml
# vectorizer:
#   batch_size: 100
#   model_kwargs:
#     model: text-embedding-3-small
#     dimensions: 256
#     encoding_format: float
# 
# retriever:
#   top_k: 2
# 
# generator:
#   model: gpt-3.5-turbo
#   temperature: 0.3
#   stream: false
# 
# text_splitter:
#   split_by: word
#   chunk_size: 400
#   chunk_overlap: 200
# ```



# test using only the document splitter
text_split = components["document_splitter"](documents)
print(text_split)




# test the whole data transformer
embeddings = data_transformer(documents)
print(embeddings)




db.register_transformer(data_transformer)
db.transformer_setups




db.transform_data(transformer=data_transformer)




keys = list(db.transformed_documents.keys())
documents = db.transformed_documents[keys[0]]
vectors = [doc.vector for doc in documents]
print(len(vectors), type(vectors), vectors[0][0:10])

# check if all embeddings are the same length
dimensions = set([len(vector) for vector in vectors])
dimensions




# check the length of all documents,text 
lengths = set([doc.estimated_num_tokens for doc in documents])
print(lengths)




total = 0
for doc in documents:
    if len(doc.vector) != 256:
        print(doc)
        total+=1
print(total)




# save the db states, including the original documents with len 1, and transformed documents
db.save_state("tutorials/db_states.pkl")




# construct the db

restored_db = LocalDB.load_state("tutorials/db_states.pkl")
restored_db




len_documents=len(restored_db.documents)
keys = list(restored_db.transformed_documents.keys())
len_transformed_documents=len(restored_db.transformed_documents[keys[0]])
print(len_documents, len_transformed_documents, keys)




# lets' print out part of the vector
restored_db.transformed_documents[keys[0]][0].vector[0:10]


# Now we have prepared the embeddings which can be used in ``FAISSRetriever``. The ``FAISSRetriever`` is a simple wrapper around the FAISS library. It is a simple and efficient way to search for the nearest neighbors in the embedding space.



from adalflow.components.retriever import FAISSRetriever



retriever = FAISSRetriever(embedder=components["embedder"], top_k=5)
print(retriever)




documents = restored_db.transformed_documents[keys[0]]
vectors = [doc.vector for doc in documents]
print(len(vectors), type(vectors), vectors[0][0:10])

# check if all embeddings are the same length
dimensions = set([len(vector) for vector in vectors])
dimensions




# convert vectors to numpy array
import numpy as np
vectors_np = np.array(vectors, dtype=np.float32)




retriever.build_index_from_documents(documents=vectors)




# retriever for a single query
query = "What happened at Viaweb and Interleaf?"
second_query = "What company did Paul Graham co-found?"

output = retriever(input=[query, second_query])
output




# get initial documents
for per_query_output in output:
    per_query_output.documents = [documents[idx] for idx in per_query_output.doc_indices]

output












# In the RAG notes, we will combine this with Generator to get the end to end response.

# ## BM25Retriever
# 



from adalflow.components.retriever import BM25Retriever

index_strings = [doc.text for doc in documents]

retriever = BM25Retriever(documents=index_strings)

# retriever.build_index_from_documents(documents=index_strings)

output = retriever(input=[query, second_query])
output




retriever = BM25Retriever(top_k=1)
retriever.build_index_from_documents(["hello world", "world is beautiful", "today is a good day"])
output = retriever.retrieve("hello")
output




# save the index

path = "tutorials/bm25_index.json"
retriever.save_to_file(path)




retriever_loaded = BM25Retriever.load_from_file(path)




# test the loaded index
output = retriever_loaded.retrieve("hello", top_k=1)
output




retriever_loaded

