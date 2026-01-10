from fastapi import FastAPI
from app.recommender import recommend  # Import the function we just wrote

app = FastAPI(
    title="Netflix Recommender API",
    description="Content-based movie/show recommendation using TF-IDF + LSA",
    version="1.0"
)

@app.get("/")
def root():
    return {"message": "Welcome to Netflix Recommender API"}

@app.get("/recommend/{title}")
def get_recommendations(title: str, top_k: int = 5):
    """
    Get top_k recommendations for a given movie/show title
    """
    return recommend(title, top_k)
