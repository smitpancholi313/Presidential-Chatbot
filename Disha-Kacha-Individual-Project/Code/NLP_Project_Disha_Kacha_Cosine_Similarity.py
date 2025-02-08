import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

file_path = "speeches.xlsx"
metadata_df = pd.read_excel(file_path)

print(metadata_df.head())
def get_most_similar_speech(query, metadata_df):
    # Convert the speeches to a list
    speeches = metadata_df['transcript'].tolist()
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(speeches)
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)
    most_similar_idx = cosine_similarities.argmax()
    similar_speech = metadata_df.iloc[most_similar_idx]
    return similar_speech, cosine_similarities[0][most_similar_idx]

query = "What was the main focus of president's speech on unity?"
similar_speech, similarity_score = get_most_similar_speech(query, metadata_df)
print(f"Most Similar Speech:\n{similar_speech['transcript']}")
print(f"Similarity Score: {similarity_score}")
