import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

data = pd.read_excel('speeches.xlsx')
documents = data['transcript'].tolist()

def retrieve_documents(query, documents, top_n=3):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_n_indices = np.argsort(cosine_similarities)[-top_n:]
    return [documents[i] for i in top_n_indices]

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_dim, embedding_dim):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_tensor, hidden):
        embedded = self.embedding(input_tensor)
        encoder_outputs, (hidden, cell) = self.encoder_lstm(embedded, hidden)
        return encoder_outputs, (hidden, cell)

    def generate(self, input_tensor, max_len=20):
        batch_size = input_tensor.size(0)  
        if batch_size == 1:
            hidden = (torch.zeros(1, hidden_dim).to(input_tensor.device),  
                      torch.zeros(1, hidden_dim).to(input_tensor.device))  
        else:
            hidden = (torch.zeros(1, batch_size, hidden_dim).to(input_tensor.device),  
                      torch.zeros(1, batch_size, hidden_dim).to(input_tensor.device)) 
        predicted_words = []

        for i in range(max_len):
            output, (hidden, cell) = self(input_tensor, hidden)
            predicted_word = torch.argmax(output[0, 0, :])  
            predicted_words.append(predicted_word.item())  
            input_tensor = predicted_word.unsqueeze(0).unsqueeze(0)  
        generated_words = [idx2word[idx] for idx in predicted_words]
        generated_sentence = " ".join(generated_words)
        return generated_sentence


word2idx = {'What': 0, 'is': 1, 'the': 2, 'president\'s': 3, 'view': 4, 'on': 5, 'unity?': 6, '<UNK>': 7, '<PAD>': 8}
idx2word = {0: 'What', 1: 'is', 2: 'the', 3: 'president\'s', 4: 'view', 5: 'on', 6: 'unity?', 7: '<UNK>', 8: '<PAD>'}

vocab_size = len(word2idx)
hidden_dim = 64
embedding_dim = 128
model = Seq2Seq(vocab_size, hidden_dim, embedding_dim)
query = "What is the president's view on unity?"
input_tensor = torch.tensor([word2idx.get(word, word2idx['<UNK>']) for word in query.split()])
max_length = 20  
input_tensor = torch.cat([input_tensor, torch.tensor([word2idx['<PAD>']] * (max_length - len(input_tensor)))], dim=0)
input_tensor = input_tensor.unsqueeze(0)
answer = model.generate(input_tensor.squeeze(0))  
print("Generated Answer:", answer)
