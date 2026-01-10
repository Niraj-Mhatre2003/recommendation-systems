import pickle
import numpy as np
import pandas as pd
from numpy.linalg import norm

# Load models ONCE
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/svd_model.pkl", "rb") as f:
    svd = pickle.load(f)

# Load data
df = pd.read_csv("data/cleaned_netflix.csv")

# Build matrices
tfidf_matrix = vectorizer.transform(df["clean_text"])
lsa_matrix = svd.transform(tfidf_matrix)

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def recommend(title: str, top_k: int = 5):
    if title not in df["title"].values:
        return {"error": "Title not found"}

    idx = df[df["title"] == title].index[0]
    scores = []

    for i in range(len(df)):
        if i != idx:
            sim = cosine_similarity(lsa_matrix[idx], lsa_matrix[i])
            scores.append((df.iloc[i]["title"], float(sim)))

    scores.sort(key=lambda x: x[1], reverse=True)

    return {
        "title": title,
        "recommendations": scores[:top_k]
    }

