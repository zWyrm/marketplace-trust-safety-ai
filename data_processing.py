import pandas as pd
import numpy as np

def preprocess_data(input_file):
    # Load data
    df = pd.read_csv(input_file)
    print(f"Original dataset shape: {df.shape}")
    
    # remove (₹ ,) symbols from price colmn
    df['discounted_price'] = df['discounted_price'].str.replace('₹', '').str.replace(',', '').astype(float)
    df['actual_price'] = df['actual_price'].str.replace('₹', '').str.replace(',', '').astype(float)
    
    # add comma-separated colmns into lists
    multi_value_cols = ['user_id', 'user_name', 'review_id', 'review_title', 'review_content']
    
    for col in multi_value_cols:
        df[col] = df[col].astype(str).str.split(',')
    
    # max list len for each row
    df['max_reviews'] = df[multi_value_cols].apply(
        lambda row: max(len(row[col]) for col in multi_value_cols), axis=1
    )
    
    # padding
    for col in multi_value_cols:
        df[col] = df.apply(
            lambda x: x[col] + [None] * (x['max_reviews'] - len(x[col])), axis=1
        )
    
    # explode all multi-value colmns simultaneously
    exploded_df = df.explode(multi_value_cols).reset_index(drop=True)
    
    # clean up
    exploded_df.drop('max_reviews', axis=1, inplace=True)
    
    # remove rows where all review fields are None
    exploded_df = exploded_df.dropna(subset=['user_id', 'user_name', 'review_content'])
    
    print(f"Processed dataset shape: {exploded_df.shape}")
    return exploded_df

if __name__ == "__main__":
    cleaned_data = preprocess_data("cleaned_data.csv")
    cleaned_data.to_csv("cleaned_data.csv", index=False)
    print("Data preprocessing complete!")
