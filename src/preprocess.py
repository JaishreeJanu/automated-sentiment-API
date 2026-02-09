import pandas as pd
import re
from sklearn.model_selection import train_test_split
import os

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<br />', ' ', text) # Remove HTML break tags
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove special characters
    return text

def preprocess():
    # 1. Load data
    df = pd.read_csv('data/IMDB Dataset.csv')
    
    # 2. Clean
    df['review'] = df['review'].apply(clean_text)
    
    # 3. Encode Label (sentiment: positive/negative -> 1/0)
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    
    # 4. Split
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    # 5. Save
    os.makedirs('data/processed', exist_ok=True)
    train.to_csv('data/processed/train.csv', index=False)
    test.to_csv('data/processed/test.csv', index=False)
    print("Preprocessing complete. Files saved in data/processed/")

if __name__ == "__main__":
    preprocess()