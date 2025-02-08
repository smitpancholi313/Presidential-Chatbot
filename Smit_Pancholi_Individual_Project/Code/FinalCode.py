import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch
import pandas as pd
from datetime import datetime, timedelta
import re
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

filepath1 = "speeches.xlsx"
filepath2 = "speeches_russian_PM.xlsx"

data1 = pd.read_excel(filepath1)
data2 = pd.read_excel(filepath2)
data2 = data2.rename(columns={"transcript_filtered": "transcript"})

data = pd.concat([data1, data2], ignore_index=True)
data['transcript'] = data['transcript'].fillna("").astype(str)

output_dir = "./fine_tuned_president_5_epochs"

fine_tuned_model = GPT2LMHeadModel.from_pretrained(output_dir).to(device)
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

if fine_tuned_tokenizer.pad_token is None:
    fine_tuned_tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    fine_tuned_model.resize_token_embeddings(len(fine_tuned_tokenizer))


model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="./chromadb_store")
collection = client.get_or_create_collection(name="speech_embeddings")

chunk_size = 100
chunks = []
chunk_ids = []
for idx, speech in enumerate(data['transcript']):
    words = speech.split()
    for i in range(0, len(words), chunk_size):
        chunk_text = " ".join(words[i:i + chunk_size])
        chunk_id = f"{idx}_{i}" 
        chunks.append(chunk_text)
        chunk_ids.append(chunk_id)
        print(f"Chunk ID: {chunk_id}")
        print(f"Chunk Text: {chunk_text[:10]}...")

embeddings = model.encode(chunks, convert_to_tensor=False)
for chunk_id, chunk_text, embedding in zip(chunk_ids, chunks, embeddings):
    collection.add(
        documents=[chunk_text],
        metadatas=[{"chunk_id": chunk_id}],
        ids=[chunk_id],
        embeddings=[embedding]
    )

def clean_text(text):
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\x00-\x7F]+', '', text)  
    text = re.sub(r'(?:(\b\w+\b)\s+)+\1\b', r'\1', text)  
    return text

def get_top_chunks(query, n_top=5):
    query_embedding = model.encode([query], convert_to_tensor=False)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_top
    )
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

##############################
# Summarization
##############################
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

def summarize_response(response):
    if len(response.split()) <= 20:  
        return response
    summarized = summarizer(response, max_length=75, min_length=10, do_sample=False)
    return summarized[0]['summary_text']

###############################
# Sentiment Analysis
###############################
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if torch.cuda.is_available() else -1)

def analyze_sentiment(response):
    sentiment = sentiment_analyzer(response)
    return sentiment[0]

###############################
# Named Entity Recognition (NER)
###############################
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", device=0 if torch.cuda.is_available() else -1, aggregation_strategy="simple")

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

###############################
# Chat History Functionality
###############################
chat_history = {}

def clean_chat_history():
    current_time = datetime.now()
    fifteen_minutes_ago = current_time - timedelta(minutes=15)
    keys_to_remove = [key for key, (timestamp, _) in chat_history.items() if timestamp < fifteen_minutes_ago]
    for key in keys_to_remove:
        del chat_history[key]

###############################
# Chat Response Handling
###############################
def get_response(question):
    clean_chat_history()
    if question in chat_history:
        timestamp, response = chat_history[question]
        print("\nResponse from history!")
        return response
    relevant_context, is_relevant = get_most_relevant_chunk(question)
    if not is_relevant:
        return "Please ask something directly related to the speeches."
    answer = answer_question(question)
    summarized_answer = summarize_response(answer)
    sentiment = analyze_sentiment(answer)
    ner_entities = extract_named_entities(answer)
    chat_history[question] = (datetime.now(), (answer, summarized_answer, sentiment, ner_entities))
    return answer, summarized_answer, sentiment, ner_entities

print("Ask a question to the President (type 'exit' to quit):")
while True:
    input_text = input("You: ")
    if input_text.lower() == "exit":
        print("Exiting... Goodbye!")
        break
    result = get_response(input_text)
    if isinstance(result, tuple):  
        answer, summarized_answer, sentiment, ner_entities  = result
        print(f"President: {answer}")
        print(f"\nSummarized Response: {summarized_answer}")
        print(f"\nSentiment Analysis: {sentiment}")
        print(f"\nNamed Entities: {ner_entities}\n")
    else:  
        print(f"President: {result}")
