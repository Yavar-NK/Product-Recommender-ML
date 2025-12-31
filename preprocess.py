import pandas as pd
import numpy as np
import os
import kagglehub
import warnings

warnings.filterwarnings('ignore')

print("Downloading dataset...")
path = kagglehub.dataset_download("samantas2020/online-retail-xlsx")
excel_file_path = os.path.join(path, 'Online Retail.xlsx')
df = pd.read_excel(excel_file_path)

print("Cleaning data...")
df.dropna(inplace=True) 
df['InvoiceNo'] = df['InvoiceNo'].astype(str)
df['StockCode'] = df['StockCode'].astype(str)
df = df[~df['InvoiceNo'].str.contains('C')] 
df = df[df["Quantity"] > 0] 

if not os.path.exists('data'):
    os.makedirs('data')
df.to_csv('data/cleaned_retail_data.csv', index=False)
print("Success: Data saved to data/cleaned_retail_data.csv")