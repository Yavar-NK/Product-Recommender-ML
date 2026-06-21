import sys
import os
# Force python to recognize the 'src' directory structure
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import mlflow
from gensim.models import Word2Vec
from data_loader import download_and_load_data
from preprocess import clean_and_split

def train_recommender():
    # Initialize and set the MLflow tracking experiment name
    mlflow.set_experiment("Product_Recommendation_Word2Vec")
    
    with mlflow.start_run():
        # Load the raw dataset and apply data engineering pipeline
        df = download_and_load_data()
        purchases_train, products_dict = clean_and_split(df)
        
        # Define Word2Vec hyperparameters
        vector_size = 100
        window = 5
        min_count = 2
        workers = 4
        epochs = 10
        
        # Log hyperparameters to MLflow for experiment tracking and monitoring
        mlflow.log_param("vector_size", vector_size)
        mlflow.log_param("window", window)
        mlflow.log_param("epochs", epochs)
        
        print("Training Word2Vec Model...")
        # Train the sequential embedding model on purchase sequences
        model = Word2Vec(sentences=purchases_train, 
                         vector_size=vector_size, 
                         window=window, 
                         min_count=min_count, 
                         workers=workers, 
                         epochs=epochs)
        
        # Calculate and log evaluation metrics (Vocabulary size) for lineage tracking
        vocab_size = len(model.wv.index_to_key)
        mlflow.log_metric("vocab_size", vocab_size)
        
        # Ensure the output directories exist and serialize artifacts
        os.makedirs("models", exist_ok=True)
        model.save("models/word2vec.model")
        with open("models/products_dict.pkl", "wb") as f:
            pickle.dump(products_dict, f)
            
        # Log serialized artifacts and registry model directly to MLflow UI
        mlflow.log_artifact("models/products_dict.pkl")
        mlflow.log_artifacts("models", artifact_path="word2vec_model")
        
        print(f"Model trained successfully. Vocabulary size: {vocab_size}")

if __name__ == "__main__":
    train_recommender()