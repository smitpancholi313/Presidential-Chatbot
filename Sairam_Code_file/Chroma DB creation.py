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

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

#  path to data
filepath = "/home/ubuntu/sairam/Project/speeches.xlsx"
data = pd.read_excel(filepath)
print(data.head())

output_dir = "/home/ubuntu/sairam/Project/fine_tuned_president_20_epochs"
fine_tuned_model = GPT2LMHeadModel.from_pretrained(output_dir).to(device)
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

# Creating the generation pipeline using the fine tuned model created earlier
fine_tuned_generator = pipeline(
    "text-generation",
    model=fine_tuned_model,
    tokenizer=fine_tuned_tokenizer,
    device=0 if torch.cuda.is_available() else -1
)


# Chunking the data using recursive character splitter with small overlap
def preprocess_and_chunk(data, column="transcript", chunk_size=1000, chunk_overlap=20):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    for speech in data[column]:
        chunks = text_splitter.split_text(speech)
        all_chunks.extend(chunks)
    return all_chunks


# Chunk the data
chunks = preprocess_and_chunk(data)

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer

# Initialize the SentenceTransformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define a wrapper for the SentenceTransformer embedding function
class SentenceTransformerWrapper:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=True).tolist()

embedding_function = SentenceTransformerWrapper(embedding_model)

# Directory where the databse is stored. Change this
vectorstore_dir = "/home/ubuntu/sairam/Project/chromadb_combined_data"

chroma_db = Chroma(
    persist_directory=vectorstore_dir,
    embedding_function=embedding_function
)


ct=0

# Add chunks one by one. Printing to see progress
for idx, chunk in enumerate(chunks):

    print(f"New chunk {ct}out of ",len(chunks))
    ct=ct+1
    chroma_db.add_texts(texts=[chunk], metadatas=[{"chunk_id": idx}], ids=[str(idx)])
