import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gensim.models import Word2Vec

app = FastAPI(title="Product Recommender API", description="Word2Vec Recommendation System")

# Load serialized model binaries and lookup dictionaries on startup
try:
    model = Word2Vec.load("models/word2vec.model")
    with open("models/products_dict.pkl", "rb") as f:
        products_dict = pickle.load(f)
except Exception as e:
    print(f"Error loading models: {e}")

class BasketInput(BaseModel):
    basket: list[str]  # Accepts a list of product StockCodes currently in user basket

@app.post("/recommend")
def get_recommendations(data: BasketInput):
    sample_basket = data.basket
    
    # Filter out input items that do not exist in the model's learned vocabulary
    valid_items = [i for i in sample_basket if i in model.wv]
    
    if not valid_items:
        raise HTTPException(status_code=400, detail="None of the items in the basket were found in the model vocabulary.")
    
    # Compute the average vector representation (centroid) of the current basket
    basket_vec = np.mean([model.wv[i] for i in valid_items], axis=0)
    
    # Query top-5 most similar product vectors using cosine similarity
    recommendations = model.wv.most_similar(positive=[basket_vec], topn=5)
    
    # Map vector search outputs back to human-readable product metadata
    result = []
    for stock_code, score in recommendations:
        description = products_dict.get(stock_code, "Unknown Product")
        result.append({
            "stock_code": stock_code,
            "description": description,
            "similarity_score": float(score)
        })
        
    return {"input_basket": sample_basket, "recommendations": result}