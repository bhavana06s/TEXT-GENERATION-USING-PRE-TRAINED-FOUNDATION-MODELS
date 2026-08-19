
from sentence_transformers
import SentenceTransformer
import faiss
import numpy as np from transformers
import pipeline
# 1. Knowledge base
documents = [ "The Eiffel Tower is located in Paris, France and was completed in 1889.","Retrieval-AugmentedGenerationcombinesdocumentretrievalwithtextgeneration.", "Pythonisapopularhigh-levelprogramminglanguageusedinAIdevelopment.", "Vectordatabasesstoreembeddingsandsupportfastsimilaritysearch." ]
#2.Embeddocuments
embed_model=SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings=embed_model.encode(documents)
#3.BuildFAISSindex
dimension=doc_embeddings.shape[1]
index=faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))
#4.Queryandretrievetop-2relevantchunks
query="WhatisRAGinAI?"
query_embedding=embed_model.encode([query])
D,I=index.search(np.array(query_embedding),k=2)
retrieved_chunks=[documents[i]foriinI[0]]
#5.Buildaugmentedpromptandgenerateanswer
context="".join(retrieved_chunks)
prompt=f"Context:{context}\nQuestion:{query}\nAnswer:"
generator=pipeline("text2text-generation",model="google/flan-t5-base")
answer=generator(prompt,max_length=60)
print("RetrievedContext:",retrieved_chunks)
print("Answer:",answer[0]["generated_text"])
