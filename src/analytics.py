import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from wordcloud import WordCloud
import numpy as np
from textblob import TextBlob
import requests
import json
import boto3
import botocore
import io
import streamlit as st

def generate_insight(data_description, visualization_type, metrics):
    """Generate business insight using LLM."""
    try:
        # Check if API key is configured
        if not st.session_state.get('api_key_configured', False):
            print("API key not configured in session state")
            return "Please configure the API key in the Prediction tab first to get AI-generated insights."
        
        api_key = st.session_state.get('candidate_api_key')
        if not api_key:
            print("API key is empty in session state")
            return "Please configure the API key in the Prediction tab first to get AI-generated insights."

        print(f"Generating insight for {visualization_type} with API key: {api_key[:5]}...")  # Log first 5 chars of API key

        prompt = f"""As a wine industry expert and data analyst, provide a brief but insightful business analysis of the following data visualization:

Visualization Type: {visualization_type}
Data Description: {data_description}
Key Metrics: {metrics}

Focus on:
1. What does this data tell us about market opportunities?
2. What actionable recommendations can be made?
3. What potential risks or areas of attention are highlighted?

Provide your analysis in 2-3 concise sentences, focusing on practical business implications."""

        print(f"Making insight API request for {visualization_type}")  # Debug log
        
        response = requests.post(
            "https://candidate-llm.extraction.artificialos.com/v1/chat/completions",
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a wine industry expert and data analyst."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            }
        )
        
        print(f"Response status: {response.status_code}")  # Debug log
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            print(f"Generated insight: {content}")  # Debug log
            return content
        else:
            error_msg = f"API error (status {response.status_code}): {response.text}"
            print(error_msg)  # Debug log
            return f"Unable to generate AI insight: {error_msg}"
            
    except Exception as e:
        error_msg = f"Error generating insight: {str(e)}"
        print(error_msg)  # Debug log
        return error_msg

def create_ratings_distribution(df):
    """Create a box plot of wine ratings by country."""
    fig = px.box(df, x='country', y='points', 
                 title='Distribution of Wine Ratings by Country',
                 labels={'points': 'Rating (Points)', 'country': 'Country'},
                 color='country')
    fig.update_layout(showlegend=False)
    
    # Calculate metrics for insight generation
    metrics = {
        'median_points': df.groupby('country')['points'].median().to_dict(),
        'point_ranges': df.groupby('country')['points'].agg(['min', 'max']).to_dict()
    }
    
    insight = generate_insight(
        "Distribution of wine ratings across different countries",
        "Box Plot",
        metrics
    )
    
    return fig, insight

def create_price_vs_points(df):
    """Create a scatter plot of price vs points."""
    fig = px.scatter(df, x='price', y='points',
                    color='country',
                    title='Quality-Price Relationship',
                    labels={'price': 'Price ($)', 'points': 'Rating (Points)'},
                    hover_data=['variety'])
    
    # Calculate metrics for insight generation
    metrics = {
        'correlation': df['price'].corr(df['points']),
        'avg_price_by_rating': df.groupby('points')['price'].mean().to_dict()
    }
    
    insight = generate_insight(
        "Relationship between wine prices and their quality ratings",
        "Scatter Plot",
        metrics
    )
    
    return fig, insight

def create_avg_price_by_country(df):
    """Create a bar chart of average prices by country."""
    avg_prices = df.groupby('country')['price'].mean().reset_index()
    fig = px.bar(avg_prices, x='country', y='price',
                 title='Average Wine Prices by Country',
                 labels={'price': 'Average Price ($)', 'country': 'Country'},
                 color='country')
    fig.update_layout(showlegend=False)
    
    metrics = {
        'avg_prices': avg_prices.set_index('country')['price'].to_dict(),
        'price_ranges': df.groupby('country')['price'].agg(['min', 'max']).to_dict()
    }
    
    insight = generate_insight(
        "Average wine prices across different countries",
        "Bar Chart",
        metrics
    )
    
    return fig, insight

def create_top_varieties(df):
    """Create a stacked bar chart of top varieties by country."""
    variety_counts = df.groupby(['country', 'variety']).size().reset_index(name='count')
    top_varieties = variety_counts.nlargest(20, 'count')
    fig = px.bar(top_varieties, x='country', y='count',
                 color='variety',
                 title='Top Wine Varieties by Country',
                 labels={'count': 'Number of Wines', 'country': 'Country'})
    
    metrics = {
        'top_varieties_by_country': df.groupby('country')['variety'].agg(lambda x: x.value_counts().head(3).to_dict()).to_dict(),
        'variety_counts': variety_counts.groupby('country')['count'].sum().to_dict()
    }
    
    insight = generate_insight(
        "Distribution of wine varieties across different countries",
        "Stacked Bar Chart",
        metrics
    )
    
    return fig, insight

def create_sentiment_analysis(df):
    """Create sentiment analysis visualization of wine descriptions."""
    df['sentiment'] = df['description'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    
    fig = px.histogram(df, x='sentiment', color='country',
                      title='Sentiment Distribution in Wine Descriptions',
                      labels={'sentiment': 'Sentiment Score', 'count': 'Number of Wines'},
                      nbins=50)
    
    metrics = {
        'avg_sentiment': df.groupby('country')['sentiment'].mean().to_dict(),
        'sentiment_ranges': df.groupby('country')['sentiment'].agg(['min', 'max']).to_dict()
    }
    
    insight = generate_insight(
        "Sentiment analysis of wine descriptions by country",
        "Histogram",
        metrics
    )
    
    return fig, insight

def create_correlation_heatmap(df):
    """Create a correlation heatmap for numerical variables."""
    corr = df[['price', 'points']].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmin=-1, zmax=1))
    
    fig.update_layout(title='Correlation Between Price and Points')
    
    metrics = {
        'correlation_coefficient': corr.iloc[0,1],
        'price_range': {'min': df['price'].min(), 'max': df['price'].max()},
        'points_range': {'min': df['points'].min(), 'max': df['points'].max()}
    }
    
    insight = generate_insight(
        "Correlation between wine prices and ratings",
        "Heatmap",
        metrics
    )
    
    return fig, insight

def create_variety_price_distribution(df):
    """Create a box plot of prices by variety."""
    top_varieties = df['variety'].value_counts().nlargest(10).index
    df_top = df[df['variety'].isin(top_varieties)]
    
    fig = px.box(df_top, x='variety', y='price',
                 title='Price Distribution by Top Wine Varieties',
                 labels={'price': 'Price ($)', 'variety': 'Variety'},
                 color='variety')
    fig.update_layout(showlegend=False,
                     xaxis={'tickangle': 45})
    
    metrics = {
        'median_prices': df_top.groupby('variety')['price'].median().to_dict(),
        'price_ranges': df_top.groupby('variety')['price'].agg(['min', 'max']).to_dict()
    }
    
    insight = generate_insight(
        "Price distribution across different wine varieties",
        "Box Plot",
        metrics
    )
    
    return fig, insight

def get_s3_data():
    """Get data from S3 bucket."""
    try:
        # Configure S3 client for public access
        s3 = boto3.client(
            's3',
            region_name='eu-north-1',
            aws_access_key_id='',
            aws_secret_access_key='',
            config=botocore.config.Config(signature_version=botocore.UNSIGNED)
        )
        
        # Get the file from S3
        obj = s3.get_object(
            Bucket='vino-voyant-wine-origin-predictor',
            Key='wine_quality_1000.csv'
        )
        
        # Read the data directly from S3 into pandas
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        return df
    except Exception as e:
        print(f"Error accessing S3: {str(e)}")
        raise

def load_and_preprocess_data():
    """Load and preprocess data for analytics."""
    try:
        # Load data from S3
        df = get_s3_data()
        
        # Basic preprocessing
        df = df.dropna()
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None 