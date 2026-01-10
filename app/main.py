from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.recommender import recommend

app = FastAPI(
    title="Netflix Recommender API",
    description="Content-based movie/show recommendation using TF-IDF + LSA",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendRequest(BaseModel):
    title: str
    top_k: int = 5

@app.post("/recommend")
def get_recommendations(req: RecommendRequest):
    return recommend(req.title, req.top_k)

