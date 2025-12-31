import os
import pickle

model_path = "models/recommender.pkl"

if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found!")
else:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")

    def suggest(product_code):
        product_code = str(product_code)
        if product_code in model:
            print("-" * 30)
            print(f"Product ID: {product_code}")
            print(f"Product Name: {model[product_code]}")
            print("Status: Active")
            print("-" * 30)
        else:
            print(f"Product code '{product_code}' not found.")

    test_code = '22423'
    suggest(test_code)