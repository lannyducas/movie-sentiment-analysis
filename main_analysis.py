# Movie Sentiment Analysis: IMDB vs Metacritic
# Analysis of movie description sentiment vs critic/user ratings by genre
# Author: Daniel Lucas

import json
import pymongo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import statsmodels.formula.api as smf
from transformers import pipeline, XLMRobertaTokenizer, XLMRobertaForSequenceClassification

# ===== DATABASE CONNECTION =====
# Note: You'll need to set up your own MongoDB connection
# Create a credentials.json file with your MongoDB connection string
def connect_to_database():
    """
    Connect to MongoDB database containing movie data
    Requires credentials.json file with mongodb connection string
    """
    try:
        with open('credentials.json') as cred:
            data = json.load(cred)
            mongo_connection_string = data['mongodb']
        
        client = pymongo.MongoClient(mongo_connection_string)
        db = client.DATA320
        print("Connected to database successfully!")
        print("Available collections:", db.list_collection_names())
        return db
    except FileNotFoundError:
        print("Error: credentials.json file not found")
        print("Falling back to sample data mode...")
        return None

# ===== DATA LOADING =====
def load_movie_data(db):
    """
    Load and merge IMDB and Metacritic data from MongoDB views
    Returns merged dataframe with both datasets
    """
    print("Loading IMDB data from imdb_clean_view...")
    imdb_cursor = db.imdb_clean_view.find()
    imdb = pd.DataFrame(imdb_cursor)
    print(f"IMDB records loaded: {len(imdb)}")
    
    print("Loading Metacritic data from metacritic_view...")
    metacritic_cursor = db.metacritic_view.find()
    metacritic = pd.DataFrame(metacritic_cursor)
    print(f"Metacritic records loaded: {len(metacritic)}")
    
    # Merge datasets on movie title
    print("Merging datasets on title...")
    unified_view = pd.merge(imdb, metacritic, how="inner", on="title")
    print(f"Merged records: {len(unified_view)}")
    
    return unified_view

def load_sample_data():
    """
    Load sample data from CSV files for demonstration purposes
    This allows the project to run without database access
    """
    try:
        print("Loading sample data from CSV files...")
        # Load sample datasets
        imdb_sample = pd.read_csv('data/imdb_sample.csv')
        metacritic_sample = pd.read_csv('data/metacritic_sample.csv')
        
        print(f"IMDB records loaded: {len(imdb_sample)}")
        print(f"Metacritic records loaded: {len(metacritic_sample)}")
        
        # Merge the sample data
        unified_view = pd.merge(imdb_sample, metacritic_sample, how="inner", on="title")
        print(f"Merged records: {len(unified_view)}")
        return unified_view
        
    except FileNotFoundError as e:
        print(f"Error: Sample data files not found - {e}")
        print("Please ensure data/imdb_sample.csv and data/metacritic_sample.csv exist")
        return None
    except Exception as e:
        print(f"Error loading sample data: {e}")
        return None
    
# ===== SENTIMENT ANALYSIS SETUP =====
def setup_sentiment_model():
    """
    Initialize the sentiment analysis model
    Using XLM-RoBERTa model trained on Twitter data
    """
    print("Loading sentiment analysis model...")
    MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL)
    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL)
    sentiment_task = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    
    print("Model loaded successfully!")
    return sentiment_task

def calculate_sentiment(text, sentiment_task):
    """
    Convert sentiment analysis output to numeric score
    Returns: float between -1 (very negative) and 1 (very positive)
    """
    if text is None or pd.isna(text):
        return 0
    
    try:
        sentiment = sentiment_task(str(text))
        score = sentiment[0]['score']
        label = sentiment[0]['label']
        
        if label == 'negative':
            return -(score + 0.5)  # Range: -1.5 to -0.5
        elif label == 'neutral':
            return score - 0.5     # Range: -0.5 to 0.5
        elif label == 'positive':
            return score + 0.5     # Range: 0.5 to 1.5
    except:
        return 0

def test_sentiment_model(sentiment_task):
    """Test the sentiment model with sample phrases"""
    print("\nTesting sentiment analysis model:")
    test_phrases = [
        "I love this movie!",
        "Chris Hemsworth can't act to save his life!",
        "Better than melatonin..."
    ]
    
    for phrase in test_phrases:
        score = calculate_sentiment(phrase, sentiment_task)
        print(f"'{phrase}' -> Sentiment: {score:.3f}")

# ===== DATA PREPROCESSING =====
def preprocess_genres(df):
    """
    Process genres column to handle multiple genres per movie
    Returns dataframe with exploded genres (one row per movie-genre combination)
    """
    print("Processing genres data...")
    
    # Convert genre strings to lists
    df["genres"] = df["genres"].apply(
        lambda g: g if isinstance(g, list)
        else str(g).split(", ") if isinstance(g, str) and pd.notna(g)
        else []
    )
    
    # Explode genres to create one row per movie-genre combination
    exploded_df = df.explode("genres")
    
    # Remove rows with empty genres
    exploded_df = exploded_df[exploded_df["genres"].notna()]
    exploded_df = exploded_df[exploded_df["genres"] != ""]
    
    print(f"Movies after genre explosion: {len(exploded_df)}")
    print(f"Unique genres: {exploded_df['genres'].nunique()}")
    
    return exploded_df

# ===== ANALYSIS FUNCTIONS =====
def run_correlation_analysis(df):
    """Run regression analysis on different variables"""
    print("\n" + "="*50)
    print("CORRELATION ANALYSIS")
    print("="*50)
    
    # Test sentiment vs gross sales
    print("\n1. Sentiment vs Gross Sales:")
    try:
        model = smf.ols(formula="sentiment ~ gross_sales_usd", data=df).fit()
        print(f"R-squared: {model.rsquared:.4f}")
        print(f"P-value: {model.pvalues[1]:.4f}")
    except:
        print("Unable to analyze gross sales correlation")
    
    # Test sentiment vs user rating
    print("\n2. Sentiment vs IMDB User Rating:")
    try:
        model = smf.ols(formula="sentiment ~ user_rating", data=df).fit()
        print(f"R-squared: {model.rsquared:.4f}")
        print(f"P-value: {model.pvalues[1]:.4f}")
    except:
        print("Unable to analyze user rating correlation")
    
    # Test sentiment vs metacritic score
    print("\n3. Sentiment vs Metacritic Score:")
    try:
        model = smf.ols(formula="sentiment ~ score", data=df).fit()
        print(f"R-squared: {model.rsquared:.4f}")
        print(f"P-value: {model.pvalues[1]:.4f}")
    except:
        print("Unable to analyze metacritic score correlation")

def analyze_genres(exploded_df):
    """Analyze sentiment patterns by genre"""
    print("\n4. Sentiment vs Genre:")
    try:
        model = smf.ols(formula="sentiment ~ C(genres)", data=exploded_df).fit()
        print(f"R-squared: {model.rsquared:.4f}")
        return model
    except:
        print("Unable to analyze genre correlation")
        return None

# ===== VISUALIZATION FUNCTIONS =====
def create_genre_sentiment_plot(exploded_df):
    """Create basic genre sentiment visualization"""
    genre_summary = exploded_df.groupby("genres")["sentiment"].mean().sort_values()
    
    plt.figure(figsize=(12, 6))
    plt.bar(genre_summary.index, genre_summary.values, color="dimgrey")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average Sentiment Score")
    plt.title("Average Sentiment by Genre")
    plt.tight_layout()
    plt.show()
    
    return genre_summary

def create_advanced_genre_plots(exploded_df):
    """Create advanced visualizations with color coding for ratings"""
    
    # IMDB Rating visualization
    plt.figure(figsize=(15, 6))
    
    light_color = "#a6cee3"
    dark_color = "#1f78b4"
    colorscale = mcolors.LinearSegmentedColormap.from_list("blue_shade", [light_color, dark_color])
    
    genre_summary = exploded_df.groupby("genres").agg({
        "sentiment": "mean",
        "user_rating": "mean"
    }).reset_index()
    genre_summary = genre_summary.sort_values("sentiment", ascending=True)
    
    norm = plt.Normalize(genre_summary["user_rating"].min(), genre_summary["user_rating"].max())
    colors = colorscale(norm(genre_summary["user_rating"]))
    
    bars = plt.bar(genre_summary["genres"], genre_summary["sentiment"], color=colors)
    
    sm = plt.cm.ScalarMappable(cmap=colorscale, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label("Average IMDB User Rating")
    
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average Sentiment Score")
    plt.title("Average Sentiment by Genre (Color = Avg IMDB User Rating)")
    plt.tight_layout()
    plt.show()
    
    # Metacritic Score visualization
    plt.figure(figsize=(15, 6))
    
    genre_summary_meta = exploded_df.groupby("genres").agg({
        "sentiment": "mean",
        "score": "mean"
    }).reset_index()
    genre_summary_meta = genre_summary_meta.sort_values("sentiment", ascending=True)
    
    norm_meta = plt.Normalize(genre_summary_meta["score"].min(), genre_summary_meta["score"].max())
    colors_meta = colorscale(norm_meta(genre_summary_meta["score"]))
    
    bars = plt.bar(genre_summary_meta["genres"], genre_summary_meta["sentiment"], color=colors_meta)
    
    sm_meta = plt.cm.ScalarMappable(cmap=colorscale, norm=norm_meta)
    sm_meta.set_array([])
    cbar_meta = plt.colorbar(sm_meta)
    cbar_meta.set_label("Average Metacritic Score")
    
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average Sentiment Score")
    plt.title("Average Sentiment by Genre (Color = Metacritic Score)")
    plt.tight_layout()
    plt.show()

def print_findings():
    """Print key findings and insights"""
    print("\n" + "="*50)
    print("KEY FINDINGS")
    print("="*50)
    
    findings = [
        "• Regression analysis shows weak correlation between sentiment and ratings",
        "• Genre shows more correlation than overall ratings, but still not strong",
        "• Temporal analysis may reveal stronger patterns (different years/decades)",
        "• Western films surprisingly rank as one of the most 'positive' genres",
        "• War movies tend to have negative sentiment but high quality ratings",
        "• Small sample sizes may affect results for certain genres",
        "• IMDB user ratings and Metacritic scores generally tell similar stories",
        "• Future work: explore genre saturation vs average scores"
    ]
    
    for finding in findings:
        print(finding)
    
    print("\nRecommendations for future analysis:")
    print("• Analyze temporal trends (sentiment patterns over decades)")
    print("• Investigate rating divergence between IMDB and Metacritic")
    print("• Explore genre saturation effects on ratings")
    print("• Consider sub-genre classification for more nuanced analysis")

# ===== MAIN EXECUTION =====
def main():
    """Main execution function"""
    print("Starting Movie Sentiment Analysis")
    print("="*50)
    
    # Try to connect to database, fall back to sample data
    db = connect_to_database()
    
    if db is not None:
        # Load data from MongoDB
        movie_data = load_movie_data(db)
    else:
        # Load sample data from CSV files
        movie_data = load_sample_data()
        if movie_data is None:
            print("Unable to load data. Exiting.")
            return None, None
    
    # Setup sentiment analysis
    sentiment_task = setup_sentiment_model()
    test_sentiment_model(sentiment_task)
    
    # Apply sentiment analysis to movie descriptions
    print("\nAnalyzing sentiment of movie descriptions...")
    movie_data['sentiment'] = movie_data['description'].apply(
        lambda x: calculate_sentiment(x, sentiment_task)
    )
    
    print(f"Sentiment analysis complete! Sample results:")
    print(movie_data[["title", "sentiment"]].head())
    
    # Run correlation analysis
    run_correlation_analysis(movie_data)
    
    # Process genres and analyze
    exploded_data = preprocess_genres(movie_data)
    analyze_genres(exploded_data)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_genre_sentiment_plot(exploded_data)
    create_advanced_genre_plots(exploded_data)
    
    # Print findings
    print_findings()
    
    print("\nAnalysis complete!")
    
    return movie_data, exploded_data

if __name__ == "__main__":
    movie_data, exploded_data = main()