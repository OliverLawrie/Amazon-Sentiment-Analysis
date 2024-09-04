import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from spacy.lang.en.stop_words import STOP_WORDS
import string

# Load the spaCy model and add the SpacyTextBlob pipeline
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

# Step 1: Load the dataset
def load_dataset():
    return pd.read_csv("amazon_product_reviews.csv")

# Step 2: Preprocess the text data by dropping na values
def preprocess_text_data(data):
    # Select the 'reviews.text' column and remove missing values
    clean_data = data.dropna(subset=['reviews.text'])
    return clean_data

# Function to preprocess text by removing stop words punctuation + lemmatizing tokens
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower().strip() for token in doc if token.text not in string.punctuation and token.text.lower().strip() not in STOP_WORDS]
    return ' '.join(tokens)

# Step 3: Define a function for sentiment analysis and get polarity
def analyze_sentiment(text):
    doc = nlp(text)
    sentiment = doc._.polarity
    return sentiment

# Step 4: Test the model on sample product reviews
def test_model(data):
    sample_reviews = data['reviews.text'][:5]  # Assuming we want to test on the first 5 reviews
    for review in sample_reviews:
        preprocessed_review = preprocess_text(review)
        sentiment = analyze_sentiment(preprocessed_review)
        print("Review:", review)
        print("Preprocessed Review:", preprocessed_review)
        print("Sentiment:", sentiment)

# Main function
def main():
    # Step 1: Load dataset
    dataset = load_dataset()
    print(dataset.head())

    # Step 2: Preprocess text data
    clean_data = preprocess_text_data(dataset)

    # Step 4: Test the model
    test_model(clean_data)

if __name__ == "__main__":
    main()