import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch
import pandas as pd
from datetime import datetime, timedelta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.text(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
filepath1 = "speeches.xlsx"
filepath2 = "speeches_russian_PM.xlsx"
data1 = pd.read_excel(filepath1)
data2 = pd.read_excel(filepath2)
data2 = data2.rename(columns={"transcript_filtered": "transcript"})
data = pd.concat([data1, data2], ignore_index=True)
data['transcript'] = data['transcript'].fillna("").astype(str)
output_dir = "./fine_tuned_president_20_epochs"
fine_tuned_model = GPT2LMHeadModel.from_pretrained(output_dir).to(device)
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

fine_tuned_generator = pipeline(
    "text-generation",
    model=fine_tuned_model,
    tokenizer=fine_tuned_tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    temperature=0.7,
    max_new_tokens=512
)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english",
                              device=0 if torch.cuda.is_available() else -1)

chunk_size = 100
chunks = []

for speech in data['transcript']:
    words = speech.split()
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(chunks)

def get_most_relevant_chunk(question, top_n=2, similarity_threshold=0.5):
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, tfidf_matrix)

    top_n_indices = np.argsort(similarities[0])[-top_n:][::-1]

    highest_similarity = similarities[0][top_n_indices[0]]
    if highest_similarity < similarity_threshold:
        return None, False

    combined_context = " ".join([chunks[idx] for idx in top_n_indices])
    return combined_context, True


def answer_question(question):
    relevant_context, is_relevant = get_most_relevant_chunk(question)

    if not is_relevant:
        return "Please ask something directly related to the speeches."

    prompt = (
        f"The following context is strictly derived from the speeches dataset:\n\n"
        f"Context: {relevant_context}\n\n"
        f"Based only on this context, answer the following question:\n"
        f"Question: {question}\nAnswer:"
    )

    response = fine_tuned_generator(
        prompt,
        max_new_tokens=512,
        num_return_sequences=1,
        pad_token_id=fine_tuned_tokenizer.eos_token_id,
        truncation=True
    )

    return response[0]['generated_text'].split('Answer:')[-1].strip()


def summarize_response(response):
    if len(response.split()) <= 20:
        return response

    summarized = summarizer(response, max_length=50, min_length=10, do_sample=False)
    return summarized[0]['summary_text']


def analyze_sentiment(response):
    sentiment = sentiment_analyzer(response)
    return sentiment[0]

st.title("Presidential Speeches NLP App")
st.write("Ask questions based on the speeches dataset.")
question = st.text_input("Enter your question:")

if question:
    st.write("Processing your question...")

    answer = answer_question(question)
    summarized_answer = summarize_response(answer)
    sentiment = analyze_sentiment(summarized_answer)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Summarized Response")
    st.write(summarized_answer)

    st.subheader("Sentiment Analysis")
    sentiment_label = sentiment['label']
    sentiment_score = round(sentiment['score'], 2)

    st.write(f"**LABEL:** {sentiment_label}")
    st.write(f"**SCORE:** {sentiment_score}")
