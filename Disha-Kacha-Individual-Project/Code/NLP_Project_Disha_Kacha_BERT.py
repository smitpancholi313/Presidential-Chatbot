import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration, BartTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large') 
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bart_model.to(device)

df = pd.read_excel('speeches.xlsx')
documents = df['transcript'].tolist()
document_embeddings = sentence_model.encode(documents, batch_size=32, show_progress_bar=True)

def retrieve_documents_with_embeddings(query, documents, top_n=5):
    query_embedding = sentence_model.encode([query])
    cosine_similarities = cosine_similarity(query_embedding, document_embeddings).flatten()
    top_n_indices = np.argsort(cosine_similarities)[-top_n:][::-1]  
    retrieved_docs = [documents[i] for i in top_n_indices]
    return retrieved_docs, cosine_similarities[top_n_indices]

def generate_answer(question, context):
    input_text = f"Answer the question based on the context provided: question: {question} context: {context}"
    print("Input text: ", input_text)
    inputs = bart_tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    output = bart_model.generate(inputs,
                                 max_length=1000,  
                                 num_beams=5,  
                                 no_repeat_ngram_size=3,  
                                 early_stopping=True)  
    answer = bart_tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    question = input("Enter a question: ")
    retrieved_documents, scores = retrieve_documents_with_embeddings(question, documents, top_n=3)  
    print("\nRetrieved Documents:")
    for i, doc in enumerate(retrieved_documents):
        print(f"Doc {i + 1} (Score: {scores[i]:.4f}): {doc[:1000]}...")  
    combined_context = " ".join(retrieved_documents)  
    answer = generate_answer(question, combined_context)
    print("\nGenerated Answer:", answer)
