import pickle
import numpy as np
import pandas as pd
from numpy.linalg import norm

# Load models
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/svd_model.pkl", "rb") as f:
    svd = pickle.load(f)

# Load cleaned data
df = pd.read_csv("data/cleaned_netflix.csv")

# Prepare LSA matrix for recommendation
tfidf_matrix = vectorizer.transform(df['clean_text'])
lsa_matrix = svd.transform(tfidf_matrix)

# Cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# Recommendation function
def recommend(title, top_k=5):
    if title not in df['title'].values:
        return {"error": "Title not found"}
    
    idx = df[df['title'] == title].index[0]
    scores = []

    for i in range(len(df)):
        if i != idx:
            sim = cosine_similarity(lsa_matrix[idx], lsa_matrix[i])
            scores.append((df.iloc[i]['title'], float(sim)))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return {"title": title, "recommendations": scores[:top_k]}

import pickle

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("models/svd_model.pkl", "rb") as f:
    svd = pickle.load(f)
import pickle
import os

MODEL_PATH = "models/tfidf_vectorizer.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    vectorizer = pickle.load(f)
