# 🛍️ Neural Product Recommendation System

An end-to-end Machine Learning pipeline that leverages **Natural Language Processing (NLP)** techniques to build a recommendation engine. By treating customer purchase sequences as "sentences," this project uses **Word2Vec** to learn latent relationships between products.

## 🚀 Interactive Demo
You can run the complete analysis, train the model, and test the recommendations directly in your browser:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1p5cUPRro3EyT1b8wAz_Em3HuAHzK82fL)

## 🛠️ Key Features
- **Neural Embeddings:** Implemented **Word2Vec (Skip-gram)** to map products into a 150-dimensional vector space.
- **Advanced Market Basket Analysis:** Used a **Weighted Strategy** to prioritize recent items in a user's basket for more accurate next-item prediction.
- **High-Dimensional Visualization:** Applied **UMAP** and **t-SNE** to visualize product clusters and identify similar item groups.
- **Robust Evaluation:** Measured performance using **Hit Rate @ 10**, ensuring the model's reliability in a real-world retail context.

## 📂 Project Structure
- `Product_Recommendations_using_word2vec.ipynb`: Full research and development notebook.
- `preprocess.py`: Data cleaning (handling missing values and cancelled orders) and sequence generation.
- `train.py`: Model training script with hyperparameter tuning (`window=5`, `epochs=30`).
- `predict.py`: Inference script for generating real-time recommendations.

## 📊 Technical Highlights
- **Dataset:** Online Retail dataset containing transactions from a UK-based non-store retail.
- **Data Engineering:** Handled 24.9% missing values in `CustomerID` and filtered out cancelled transactions (InvoiceNo starting with 'C') to ensure data quality.
- **Optimization:** Tuned the Word2Vec model with `negative sampling=15` and `min_count=2` to capture meaningful associations even for less frequent items.

## 📂 Structure & 🚀 Quick Start

### Project Organization
- `Product_Recommendations_using_word2vec.ipynb`: Full R&D notebook.
- `preprocess.py` & `train.py`: Data cleaning and model training logic.
- `predict.py`: Real-time inference script.

### How to Run
1. **Clone & Install:**
   ```bash
   git clone [https://github.com/Yavar-NK/Product-Recommender-ML.git](https://github.com/Yavar-NK/Product-Recommender-ML.git)
   pip install -r requirements.txt
