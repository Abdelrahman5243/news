from flask import Flask, jsonify
import pandas as pd
import os
import sys
import csv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from threading import Thread
import time
from datetime import datetime, timedelta
from newsapi import NewsApiClient
import re

# Initialize NewsApiClient with your API key
api_key = 'aa138deb80a44635a21b89f2e060d2f9'
newsapi = NewsApiClient(api_key=api_key)

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def save_news_data():
    try:
        # Get the current year
        current_year = datetime.now().year

        # Get the directory path of the current script
        current_directory = os.path.dirname(os.path.abspath(__file__))

        # Calculate the date range for the last 7 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        # Define keywords and sources for news articles
        keywords = {
            'apple': ['apple stock price news'],
            'microsoft': ['Microsoft stock price news'],
            'google': ['Google stock price news'],
            'forex': ['forex Egypt', 'Egyptian pound exchange rate', 'forex market Egypt', 'Egypt economy', 'dollar exchange rate in Egypt', 'currency conversion in Egypt',
                    'Egyptian economy news', 'Egyptian currency updates', 'Egyptian financial market', 'Cairo stock exchange', 'Egyptian GDP growth', 'Egyptian inflation rate',
                    'Egyptian monetary policy', 'Central Bank of Egypt'],
        }

        # Fetch news articles based on keywords and sources for each company
        for company, keyword_list in keywords.items():
            articles = []
            for keyword in keyword_list:
                news = newsapi.get_everything(q=keyword, language='en', page_size=100, from_param=start_date.strftime('%Y-%m-%d'), to=end_date.strftime('%Y-%m-%d'))
                articles.extend(news['articles'])

            # Extract relevant data from news articles
            data = {
                'date': [article['publishedAt'] for article in articles],
                'content': [clean_text(article['content']) for article in articles],  # Clean the content
                'headlines': [article['title'] for article in articles]
            }

            # Filter articles that are from the current year
            data_filtered = {
                'date': [],
                'content': [],
                'headlines': []
            }
            for i in range(len(data['date'])):
                article_year = datetime.strptime(data['date'][i], '%Y-%m-%dT%H:%M:%SZ').year
                if article_year == current_year:
                    data_filtered['date'].append(data['date'][i])
                    data_filtered['content'].append(data['content'][i])
                    data_filtered['headlines'].append(data['headlines'][i])

            # Create a DataFrame from the filtered data
            df = pd.DataFrame(data_filtered)

            # Convert 'date' column to datetime
            df['date'] = pd.to_datetime(df['date'])

            # Remove duplicate rows based on content
            df.drop_duplicates(subset=['content'], inplace=True)

            # Sort the DataFrame by date
            df = df.sort_values(by='date')

            # Define the filename based on the company
            filename = f'{company}_data.csv'

            # Construct the file path to save the CSV file in the function's directory
            file_path = os.path.join(current_directory, filename)

            # Save the DataFrame to a CSV file
            df.to_csv(file_path, index=False)

            print(f"{company.capitalize()} news data saved successfully in {file_path}.")

    except Exception as e:
        print("Error occurred:", str(e))


app = Flask(__name__)

# Download NLTK resources (if not already downloaded)
nltk.download('vader_lexicon')

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    # Perform sentiment analysis
    sentiment_scores = sia.polarity_scores(text)

    # Determine sentiment label based on compound score
    if sentiment_scores['compound'] > 0.05:
        return "Positive"
    elif sentiment_scores['compound'] < -0.05:
        return "Negative"
    else:
        return "Neutral"

def process_csv(input_filename, output_filename):
    try:
        if not os.path.exists(input_filename):
            print(f"Error: Input file '{input_filename}' not found.")
            return

        with open(input_filename, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            data = list(reader)

        for item in data:
            # Analyze sentiment for the content
            item['predicted_sentiment'] = analyze_sentiment(item['content'])

        # Write the results to a new CSV file
        fieldnames = ['date', 'content', 'predicted_sentiment']
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in data:
                writer.writerow({key: item[key] for key in fieldnames})

        print(f"Sentiment analysis results saved to {output_filename}")
    
    except Exception as e:
        print(f"An error occurred while processing {input_filename}: {e}")

def make_process_csv():
    try:
        # Update file paths
        input_files = {
            'apple': 'apple_data.csv',
            'microsoft': 'microsoft_data.csv',
            'google': 'google_data.csv',
            'forex': 'forex_data.csv'
        }

        output_files = {
            'apple': 'apple_sentiment.csv',
            'microsoft': 'microsoft_sentiment.csv',
            'google': 'google_sentiment.csv',
            'forex': 'forex_sentiment.csv'
        }

        # Process each CSV file
        for key in input_files:
            input_filename = input_files[key]
            output_filename = output_files[key]
            process_csv(input_filename, output_filename)
            print(f"Processed {input_filename} -> {output_filename}")
            
    except Exception as e:
        print("An error occurred during sentiment analysis:", e)

def count_sentiments(input_filename):
    try:
        # Check if input file exists
        if not os.path.exists(input_filename):
            print(f"Error: Input file '{input_filename}' not found.")
            return

        # Read CSV file into a DataFrame
        df = pd.read_csv(input_filename)

        # Count sentiments
        sentiment_counts = df['predicted_sentiment'].value_counts().to_dict()

        return sentiment_counts
    
    except Exception as e:
        print(f"An error occurred while counting sentiments in {input_filename}: {e}")
        return {}

def result():
    save_news_data()
    make_process_csv()
    sentiment_counts = {}
    for key in ['apple', 'google', 'microsoft', 'forex']:
        sentiment_counts[key] = count_sentiments(f'{key}_sentiment.csv')

    result_list = []
    for source, counts in sentiment_counts.items():
        result_list.append({
            "source": source + '-news',
            "sentiment_counts": counts
        })

    return {
        "news_analysis_last_week": result_list
    }

# Define a function to refresh the data every 6 hours
def refresh_data():
    global cached_data
    while True:
        try:
            cached_data = result()
            # Refresh data every 6 hours
            time.sleep(6 * 60 * 60)
        except Exception as e:
            print("An error occurred while refreshing data:", e)

# Preload data initially
cached_data = result()

# Start a separate thread to refresh the data
refresh_thread = Thread(target=refresh_data)
refresh_thread.daemon = True
refresh_thread.start()

@app.route('/')
def index():
    # Return cached data
    return jsonify(cached_data)

if __name__ == '__main__':
    app.run(debug=False)
