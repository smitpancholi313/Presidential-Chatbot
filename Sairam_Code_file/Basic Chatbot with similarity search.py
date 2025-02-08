#%% Imports and Setup
import os
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, pipeline
from torch.utils.data import DataLoader, Dataset
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

#%%

# Loading the Dataset
filepath = "/home/ubuntu/sairam/Project/speeches.xlsx"
data = pd.read_excel(filepath)
print(data.head())

#%%
print(data['transcript'])

#%%
# This is a base model, no vector store just simple cosine similarity. This will be used to benchmark

import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

chunk_size = 300
chunks = []

for speech in data['transcript']:
    words = speech.split()
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

# Using basic vectors right now, can update to embeddings
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(chunks)

# Model specifically trained for Q and A
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)


# Cosing similarity
# Cosing similarity
def get_most_relevant_chunk(question):
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, tfidf_matrix)

    # Get indices of the top 10 most similar chunks
    top_10_indices = np.argsort(similarities[0])[-10:][::-1]

    # Combine the top 10 most relevant chunks into one context
    combined_context = " ".join([chunks[idx] for idx in top_10_indices])

    return combined_context


# Chatbot function
def chatbot():
    print("Ask a Presedent! Type any question or statement, and the bot repsonds based on the views of past predents. Write 'exit' to quit.")
    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break

        # Retrieve the most relevant chunk
        relevant_context = get_most_relevant_chunk(user_query)

        # Use QA pipeline to answer the question
        try:
            result = qa_pipeline({"question": user_query, "context": relevant_context})
            print(f"Bot: {result['answer']}")
        except Exception as e:
            print(f"Bot: Sorry, I couldn't understand your question. (Error: {e})")

chatbot()
