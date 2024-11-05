#!/usr/bin/env python
# coding: utf-8

# Let us use OpenAIClient and `text-embedding-3-small` as an example for embedder



from adalflow.core.embedder import Embedder
from adalflow.components.model_client import OpenAIClient
from adalflow.utils import setup_env # ensure you setup OPENAI_API_KEY in your project .env file

model_kwargs = {
    "model": "text-embedding-3-small",
    "dimensions": 256,
    "encoding_format": "float",
}

query = "What is the capital of China?"

queries = [query] * 100


embedder = Embedder(model_client=OpenAIClient(), model_kwargs=model_kwargs)




print(embedder)




# call the embedder with a single query, takes around 0.3 seconds for one query
response = embedder(query)
print(response)




# dimension
print(response.embedding_dim, response.length, response.is_normalized)


# Use batch processing. It can handle 10-100 embeddings at a time, depending on the model. When we are using cloud API provider like OpenAI, we do not have clarity on how they are processing the backend or how this will impact the cost. Also, each API provider might have different pricing models.



# call the embedder with a list of queries, takes around 0.9 seconds fpr 100 queries, 2.5s for 1000 queries
response = embedder(queries)
print(response)




response.length, response.embedding_dim, response.is_normalized


# Use local model, we use ``TransformersClient`` as an example. And we will enable our library logging to see the process better.



from adalflow.core.embedder import Embedder
from adalflow.components.model_client import TransformersClient
# from adalflow.utils import enable_library_logging

# enable_library_logging(level="DEBUG")

model_kwargs = {"model": "thenlper/gte-base"}
local_embedder = Embedder(model_client=TransformersClient(), model_kwargs=model_kwargs)




print(local_embedder)




# single query, takes around 0.1 seconds for one query, this might differs on the hardware you use
response = local_embedder(query)
print(response.length)
print(response)




# multiple queries, takes around 0.7s for 100 queries
response = local_embedder(queries)




print(response.length, response.embedding_dim)




print(response)




response.data[1]


# It is a good practise to set up a maximum ``batch_size`` before calling the ``Embedder``.



from tqdm import tqdm

batch_size = 100
all_queries = [query] * 1024

for i in tqdm(range(0, len(all_queries), batch_size)):
    print(f"Processing batch {i // batch_size}")
    response = local_embedder(all_queries[i : i + batch_size])
    print(response.length)


# Use our ``BatchEmbedder`` to handle the batch processing.



from adalflow.core.embedder import BatchEmbedder

batch_embedder = BatchEmbedder(embedder=local_embedder, batch_size=100)

response = batch_embedder(all_queries)


# Check if an embedding is normalized and how to normalize it if it is not.
# Use post progressor to shrink the dimension of an embedding.



from adalflow.core.types import Embedding
from adalflow.core.functional import normalize_vector
from typing import List
from adalflow.core.component import Component
from copy import deepcopy
class DecreaseEmbeddingDim(Component):
    def __init__(self, old_dim: int, new_dim: int,  normalize: bool = True):
        super().__init__()
        self.old_dim = old_dim
        self.new_dim = new_dim
        self.normalize = normalize
        assert self.new_dim < self.old_dim, "new_dim should be less than old_dim"

    def call(self, input: List[Embedding]) -> List[Embedding]:
        output: List[Embedding] = deepcopy(input)
        for embedding in output:
            old_embedding = embedding.embedding
            new_embedding = old_embedding[: self.new_dim]
            if self.normalize:
                new_embedding = normalize_vector(new_embedding)
            embedding.embedding = new_embedding
        return output
    
    def _extra_repr(self) -> str:
        repr_str = f"old_dim={self.old_dim}, new_dim={self.new_dim}, normalize={self.normalize}"
        return repr_str


# Let us decrease the dimension of local embeddings using output_processors in Embedder.



local_embedder_256 = Embedder(
    model_client=TransformersClient(),
    model_kwargs=model_kwargs,
    output_processors=DecreaseEmbeddingDim(768, 256),
)




print(local_embedder_256)




response = local_embedder_256(query)
print(response.length, response.embedding_dim)

