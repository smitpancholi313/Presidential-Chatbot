import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch
from sentence_transformers import SentenceTransformer
import chromadb
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

output_dir = "./fine_tuned_president_5_epochs"
fine_tuned_model = GPT2LMHeadModel.from_pretrained(output_dir).to(device)
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

if fine_tuned_tokenizer.pad_token is None:
    fine_tuned_tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    fine_tuned_model.resize_token_embeddings(len(fine_tuned_tokenizer))

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="./chromadb_store")
collection = client.get_or_create_collection(name="speech_embeddings")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if torch.cuda.is_available() else -1)
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", device=0 if torch.cuda.is_available() else -1, aggregation_strategy="simple")

def clean_text(text):
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\x00-\x7F]+', '', text)  
    text = re.sub(r'(?:(\b\w+\b)\s+)+\1\b', r'\1', text)  
    return text

def get_top_chunks(query, n_top=5):
    query_embedding = sentence_model.encode([query], convert_to_tensor=False)
    results = collection.query(query_embeddings=query_embedding, n_results=n_top)
    top_chunks = [clean_text(doc) for doc in results['documents'][0]]
    top_score = results['distances'][0][0] if results['distances'] else 0.0
    return top_chunks, top_score

def get_most_relevant_chunk(question, top_n=5, similarity_threshold=0.2):
    top_chunks, top_score = get_top_chunks(question, n_top=top_n)
    if not top_chunks or len(top_chunks) == 0 or top_score < similarity_threshold:  
        return None, False
    combined_context = " ".join(set(top_chunks))
    return combined_context[:1000], True 

def remove_repeated_phrases(response):
    sentences = response.split('. ')
    seen = set()
    filtered_sentences = []
    for sentence in sentences:
        cleaned_sentence = clean_text(sentence)
        if cleaned_sentence not in seen:
            filtered_sentences.append(cleaned_sentence)
            seen.add(cleaned_sentence)
    return '. '.join(filtered_sentences)

def answer_question(question):
    relevant_context, is_relevant = get_most_relevant_chunk(question)
    if not is_relevant:
        return "I'm sorry, I couldn't find a direct match in the speeches. Please try rephrasing your question or asking something more specific."
    truncated_context = relevant_context[:1000]
    prompt = (
        f"The following context is strictly derived from the speeches dataset:\n\n"
        f"Context: {truncated_context}\n\n"
        f"Based only on this context, answer the following question:\n"
        f"Question: {question}\nAnswer:"
    )
    input_ids = fine_tuned_tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        output = fine_tuned_model.generate(
            input_ids,
            max_new_tokens=256,
            temperature=0.7,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=fine_tuned_tokenizer.eos_token_id
        )
    response = fine_tuned_tokenizer.decode(output[0], skip_special_tokens=True)
    if 'Answer:' in response:
        response = response.split('Answer:')[-1].strip()
    response = remove_repeated_phrases(response)
    return response

def summarize_response(response):
    if len(response.split()) <= 20:  
        return response
    summarized = summarizer(response, max_length=75, min_length=10, do_sample=False)
    return summarized[0]['summary_text']

def analyze_sentiment(response):
    sentiment = sentiment_analyzer(response)
    return sentiment[0]

def extract_named_entities(text):
    ner_results = ner_pipeline(text)
    entities = []
    seen = set()
    for entity in ner_results:
        word = entity['word'].strip()
        if word.startswith("##") or "#" in word:
            continue
        if word not in seen:
            entities.append({
                "entity": entity['entity_group'] if 'entity_group' in entity else entity['entity'],
                "word": word,
                "score": entity['score']
            })
            seen.add(word)
    return entities

st.title("Ask a Question to the President")
st.write("Query the speeches dataset and get insightful responses.")

question = st.text_input("Enter your question:")
if question:
    with st.spinner("Fetching response..."):
        relevant_context, is_relevant = get_most_relevant_chunk(question)
        if not is_relevant:
            st.error("Please ask something directly related to the speeches.")
        else:
            answer = answer_question(question)
            summarized_answer = summarize_response(answer)
            sentiment = analyze_sentiment(answer)
            ner_entities = extract_named_entities(answer)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Summarized Response")
            st.write(summarized_answer)

            st.subheader("Sentiment Analysis")
            st.json(sentiment)

            st.subheader("Named Entities")
            st.json(ner_entities)
