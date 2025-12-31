import pandas as pd
import os
import pickle

# ۱. بارگذاری داده‌های تمیز شده
if not os.path.exists('data/cleaned_retail_data.csv'):
    print("Error: Cleaned data not found! Run preprocess.py first.")
else:
    print("Loading data...")
    df = pd.read_csv('data/cleaned_retail_data.csv')
    
    # ۲. ساخت یک نقشه از کالاها
    print("Creating recommendation model...")
    recommendation_map = df.groupby('StockCode')['Description'].first().to_dict()
    
    # ۳. ذخیره در پوشه models
    if not os.path.exists('models'):
        os.makedirs('models')
        
    with open('models/recommender.pkl', 'wb') as f:
        pickle.dump(recommendation_map, f)
        
    print("Success: Model saved to models/recommender.pkl")