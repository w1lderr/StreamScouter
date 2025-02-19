# Import necessary libraries, functions, and variables

from typing import Union
from fastapi import FastAPI
from util import model
from util import desc_embeddings
from util import df
from util import recommend_movie
from util import display_recommendations

app = FastAPI()

@app.get("/getRecommendations")
async def get_recommendations(movie_description: str):
    recommendations = recommend_movie(movie_description, model, desc_embeddings, df, top_n=3)
    return display_recommendations(recommendations)