import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
import pandas as pd
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer

filepath1 = "speeches.xlsx"
filepath2 = "speeches_russian_PM.xlsx"
data1 = pd.read_excel(filepath1)
data2 = pd.read_excel(filepath2)
data2 = data2.rename(columns={"transcript_filtered": "transcript"})
data = pd.concat([data1, data2], ignore_index=True)
data['transcript'] = data['transcript'].fillna("").astype(str)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

def preprocess_and_chunk(data, column="transcript", chunk_size=1000, chunk_overlap=20):
 text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
 all_chunks = []
 for speech in data[column]:
  chunks = text_splitter.split_text(speech)
  all_chunks.extend(chunks)
 return all_chunks

chunks = preprocess_and_chunk(data)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class SentenceTransformerWrapper:
 def __init__(self, model):
  self.model = model
 def embed_documents(self, texts):
  return self.model.encode(texts, convert_to_tensor=True).tolist()
 def embed_query(self, text):
  return self.model.encode(text, convert_to_tensor=True).tolist()

embedding_function = SentenceTransformerWrapper(embedding_model)
vectorstore_dir = "./chromadb_combined_data"
chroma_db = Chroma(
 persist_directory=vectorstore_dir,
 embedding_function=embedding_function
)

ct=0
for idx, chunk in enumerate(chunks):
 print(f"New chunk {ct}out of ",len(chunks))
 ct=ct+1
 chroma_db.add_texts(texts=[chunk], metadatas=[{"chunk_id": idx}],ids=[str(idx)])
