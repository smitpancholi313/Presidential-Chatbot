import streamlit as st
import pinecone
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore
from groq import Groq
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch
import numpy as np

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch
import pandas as pd
from datetime import datetime, timedelta
import re
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

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

if fine_tuned_tokenizer.pad_token is None:
    fine_tuned_tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    fine_tuned_model.resize_token_embeddings(len(fine_tuned_tokenizer))

model = SentenceTransformer('all-MiniLM-L6-v2')

chunk_size = 100
chunks = []

for speech in data['transcript']:
    words = speech.split()
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))


def clean_text(text):
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'(?:(\b\w+\b)\s+)+\1\b', r'\1', text)
    return text


chunk_embeddings_np = model.encode(chunks, convert_to_tensor=False)


def get_top_chunks(query, n_top=5):
    query_embedding = model.encode([query], convert_to_tensor=False)
    scores = cosine_similarity(query_embedding, chunk_embeddings_np)[0]
    ranked_chunks = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    top_chunks = [clean_text(chunk) for _, chunk in ranked_chunks[:n_top]]
    top_score = ranked_chunks[0][0] if ranked_chunks else 0.0

    return top_chunks, top_score


def get_most_relevant_chunk(question, top_n=5, similarity_threshold=0.2):
    """
    Retrieve the top `top_n` most relevant chunks for the given question.
    """
    top_chunks, top_score = get_top_chunks(question, n_top=top_n)
    if not top_chunks or len(top_chunks) == 0 or top_score < similarity_threshold:
        return None, False

    combined_context = " ".join(set(top_chunks))
    return combined_context[:1000], True


def remove_repeated_phrases(response):
    """
    Removes overly repeated phrases or sentences in the generated response.
    """
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

    print(f"Prompt length: {len(prompt)}")

    encoding = fine_tuned_tokenizer(
        prompt,
        return_tensors='pt',
        padding=True,
        truncation=True
    ).to(device)

    with torch.no_grad():
        output = fine_tuned_model.generate(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=fine_tuned_tokenizer.pad_token_id
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

sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english",
                              device=0 if torch.cuda.is_available() else -1)


def analyze_sentiment(response):
    """
    Perform sentiment analysis on the response.
    """
    sentiment = sentiment_analyzer(response)
    return sentiment[0]


###############################
# Named Entity Recognition (NER)
###############################

ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    device=0 if torch.cuda.is_available() else -1,
    aggregation_strategy="simple"
)


def extract_named_entities(text):
    """
    Extract Named Entities (NER) from the text.
    Deduplicate entities and return a clean list.
    """
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
    """
    Remove entries in chat history that are older than 15 minutes.
    """
    current_time = datetime.now()
    fifteen_minutes_ago = current_time - timedelta(minutes=15)
    keys_to_remove = [key for key, (timestamp, _) in chat_history.items() if timestamp < fifteen_minutes_ago]
    for key in keys_to_remove:
        del chat_history[key]


###############################
# Chat Response Handling
###############################

def get_response(question):
    """
    Handle chat history, summarization, and sentiment analysis for a question.
    """
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


# ###############################
# # Interactive Chatbot
# ###############################
#
# print("Ask a question to the President (type 'exit' to quit):")
#
# while True:
#     input_text = input("You: ")
#     if input_text.lower() == "exit":
#         print("Exiting... Goodbye!")
#         break
#
#     result = get_response(input_text)
#
#     if isinstance(result, tuple):
#         answer, summarized_answer, sentiment, ner_entities = result
#         print(f"President: {answer}")
#         print(f"\nSummarized Response: {summarized_answer}")
#         print(f"\nSentiment Analysis: {sentiment}")
#         print(f"\nNamed Entities: {ner_entities}\n")
#     else:
#         print(f"President: {result[0]}")
#         print(f"\nSummarized Response: {result[1]}")
#         print(f"\nSentiment Analysis: {result[2]}")
#         print(f"\nNamed Entities: {result[3]}\n")

def get_relevant_excerpts(user_question, docsearch):
    relevent_docs = docsearch.similarity_search(user_question)
    return '\n\n------------------------------------------------------\n\n'.join(
        [doc.page_content for doc in relevent_docs[:3]]
    )


def presidential_speech_chat_completion(client, model, user_question, relevant_excerpts, additional_context):
    system_prompt = '''
    You are a presidential historian. Given the user's question and relevant excerpts from 
    presidential speeches, answer the question by including direct quotes from presidential speeches. 
    When using a quote, cite the speech that it was from (ignoring the chunk).
    '''

    if additional_context:
        system_prompt += f"\nThe user has provided this additional context:\n{additional_context}"

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"User Question: {user_question}\n\nRelevant Speech Excerpt(s):\n\n{relevant_excerpts}",
            },
        ],
        model=model,
    )

    return chat_completion.choices[0].message.content

###############################
# Main Application
###############################

def main():
    groq_api_key = "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"
    pinecone_api_key = "pcsk_2ZLJUA_MTwTAcBUJBfmZASEWfWjuhXvEjQPkCMurzPCXwR9VsCzut6nM3a3Q4UcSyfSXnj"

    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)

    # Check if the index exists
    index_name = "presidential-speeches"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # Adjust to match the embedding model dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # Initialize PineconeVectorStore
    docsearch = PineconeVectorStore(index_name=index_name, embedding=embedding_function,
                                    pinecone_api_key=pinecone_api_key)

    client = Groq(api_key=groq_api_key)

    st.image('groqcloud_darkmode.png')
    st.title("Presidential Speeches RAG")
    st.markdown("""
        Welcome! Ask questions about U.S. presidents, like "What were George Washington's views on democracy?" or "What did Abraham Lincoln say about national unity?". 
        The app matches your question to relevant excerpts from presidential speeches and generates a response using a pre-trained model.
    """)

    additional_context = st.sidebar.text_input('Additional summarization context (optional):')
    model_choice = st.sidebar.selectbox('Choose a model', ['llama3-8b-8192', 'gemma-7b-it',
                                                           './fine_tuned_president_20_epochs'])

    if model_choice == './fine_tuned_president_20_epochs':
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

    user_question = st.text_input("Ask a question about a US president:")

    if user_question:
        relevant_excerpts = get_relevant_excerpts(user_question, docsearch)
        if model_choice == './fine_tuned_president_20_epochs':
            response = get_response(user_question)  # Use custom model
        else:
            response = presidential_speech_chat_completion(client, model_choice, user_question, relevant_excerpts,
                                                           additional_context)
        st.write(response)


if __name__ == "__main__":
    main()
