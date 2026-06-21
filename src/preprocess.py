import pandas as pd

def clean_and_split(df):
    # Drop rows with missing values
    df.dropna(inplace=True)
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df['StockCode'] = df['StockCode'].astype(str)
    
    # Remove canceled orders (Invoice numbers containing 'C') and negative quantities
    df = df[~df['InvoiceNo'].str.contains('C')]
    df = df[df["Quantity"] > 0]
    
    # Sort dataset by InvoiceDate to maintain temporal order
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df.sort_values('InvoiceDate', inplace=True)
    
    # Perform a 90/10 temporal split for training and validation data
    split_date = df['InvoiceDate'].quantile(0.9)
    train_df = df[df['InvoiceDate'] <= split_date]
    
    # Create a mapping dictionary from StockCode to its text Description
    products_dict = train_df[['StockCode', 'Description']].drop_duplicates('StockCode').set_index('StockCode')['Description'].to_dict()
    
    # Group products by Customer and Invoice to create historical purchase baskets
    purchases_train = train_df.groupby(['CustomerID', 'InvoiceNo'])['StockCode'].apply(list).tolist()
    
    # Exclude single-item baskets as they do not provide co-occurrence patterns
    purchases_train = [s for s in purchases_train if len(s) > 1]
    
    return purchases_train, products_dict