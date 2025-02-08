import streamlit as st
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

filepath = "/home/ubuntu/sairam/Project/speeches.xlsx"
data = pd.read_excel(filepath)

output_dir = "/home/ubuntu/sairam/Project/fine_tuned_president_20_epochs"
fine_tuned_model = GPT2LMHeadModel.from_pretrained(output_dir).to(device)
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

fine_tuned_generator = pipeline(
    "text-generation",
    model=fine_tuned_model,
    tokenizer=fine_tuned_tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

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

# Using the chroma database which was created earlier. Currently uses recursive character splitter
embedding_function = SentenceTransformerWrapper(embedding_model)
vectorstore_dir = "/home/ubuntu/sairam/Project/chromadb_store"

chroma_db = Chroma(
    persist_directory=vectorstore_dir,
    embedding_function=embedding_function
)

def get_relevant_context(question, top_n=5):
    docs = chroma_db.similarity_search(question, k=top_n)
    combined_context = " ".join([doc.page_content for doc in docs])
    return combined_context

def answer_question(question, max_new_tokens=300):
    relevant_context = get_relevant_context(question)
    prompt = f"Context: {relevant_context}\n\nQuestion: {question}\nAnswer:"
    response = fine_tuned_generator(
        prompt,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        pad_token_id=fine_tuned_tokenizer.eos_token_id,
        truncation=True
    )
    return response[0]['generated_text'].split('Answer:')[-1].strip()

# Basic UI for now
st.title("Ask the President")
st.write("Interact with a fine-tuned GPT-2 model trained on presidential speeches. Type a question below to get an answer.")

# Input box for the user question
user_question = st.text_input("Enter your question:")

if user_question:
    answer = answer_question(user_question)
    st.write("### President's Answer:")
    st.write(answer)
