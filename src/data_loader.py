import os
import socket
import kagglehub
import pandas as pd

socket.setdefaulttimeout(300)
def download_and_load_data():
    print("Downloading dataset from Kaggle (This might take a few minutes)...")
    
    path = kagglehub.dataset_download("samantas2020/online-retail-xlsx")
    excel_file_path = os.path.join(path, 'Online Retail.xlsx')
    
    print("Loading Excel file into memory...")
    df = pd.read_excel(excel_file_path)
    return df