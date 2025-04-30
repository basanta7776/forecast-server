import pandas as pd
import numpy as np
from surprise import dump
import pickle
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import pytz  # Import pytz for timezone handling

# Load the data for recommendation
sales_file = 'data/AI Recommendation Data Structure - Sales.csv'
catalog_file = 'data/AI Recomendation Data Structure - Product Catalog - Product Catalog.csv'

# Load data
sales_df = pd.read_csv(sales_file)
product_catalog_df = pd.read_csv(catalog_file)

# Load the trained SVD model
model_file = 'model_1point025'

# Function to load the saved SVD model
def load_svd_model():
    _, svd_model = dump.load(model_file)
    return svd_model

svd_model = load_svd_model()  # Load the trained SVD model when the app starts

# Helper function to get product names by product_id
def get_product_name(product_id):
    product_info = product_catalog_df[product_catalog_df['product_id'] == product_id]
    if not product_info.empty:
        return product_info['product_name'].values[0]
    return "Unknown Product"

def get_cycle_products(customer_id, sales_df):
    # Convert 'purchased_date_time' to a datetime column if not already
    sales_df['purchased_date_time'] = pd.to_datetime(sales_df['purchased_date_time'])
 
    # Sort the data by customer_id, product_id, and purchased_date_time
    sales_df.sort_values(by=['customer_id', 'product_id', 'purchased_date_time'],inplace=True)
 
    # Calculate the time difference (in days) between consecutive purchases of the same product by the same customer
    sales_df['time_diff'] = sales_df.groupby(['customer_id', 'product_id'])['purchased_date_time'].diff().dt.days
 
    # Drop NA values in time_diff column (first purchase has no previous purchase to compare with)
    sales_df = sales_df.dropna(subset=['time_diff'])
 
    # Define a threshold for cyclic purchases (e.g., monthly cycle between 25 and 35 days)
    cyclic_threshold_month = (25, 35)
 
    # Define a threshold for cyclic purchases (e.g., weekly cycle between 7 and 10 days)
    cyclic_threshold_week = (7, 10)
 
    # Identify cyclic purchases based on monthly and weekly cycles
    monthly_cyclic_purchases = sales_df[(sales_df['time_diff'] >= cyclic_threshold_month[0]) & (sales_df['time_diff'] <= cyclic_threshold_month[1])]
    weekly_cyclic_purchases = sales_df[(sales_df['time_diff'] >= cyclic_threshold_week[0]) & (sales_df['time_diff'] <= cyclic_threshold_week[1])]
 
    # Filter the cyclic purchases for the specific customer_id for both monthly and weekly cycles
    customer_monthly_cyclic_purchases = monthly_cyclic_purchases[monthly_cyclic_purchases['customer_id'] == customer_id]
    customer_weekly_cyclic_purchases = weekly_cyclic_purchases[weekly_cyclic_purchases['customer_id'] == customer_id]
 
    # Get the unique product names or ids that were bought cyclically by this customer
    monthly_cyclic_product_ids = customer_monthly_cyclic_purchases['product_id'].unique().tolist()
    weekly_cyclic_product_ids = customer_weekly_cyclic_purchases['product_id'].unique().tolist()
 
    # Fetch the product names for the cyclic purchased products
    monthly_cyclic_product_names = [product_id for product_id in monthly_cyclic_product_ids]
    weekly_cyclic_product_names = [product_id for product_id in weekly_cyclic_product_ids]
 
    return {
        "monthly_cyclic_products": monthly_cyclic_product_names,
        "weekly_cyclic_products": weekly_cyclic_product_names
    }

# Function to recommend products based on customer_id using the trained SVD model
def recommend_products_for_customer(customer_id):
    # Filter the data for the specific customer
    customer_data = sales_df[sales_df['customer_id'] == customer_id]
    
    if customer_data.empty:
        return {"message": "Customer ID not found in the dataset.", "recommendations": []}
    
    # Extract the unique product IDs the customer has interacted with
    purchased_product_ids = customer_data['product_id'].unique()
    
    # Use the SVD model to predict ratings for other products that the customer hasn't purchased yet
    all_product_ids = sales_df['product_id'].unique()
    recommended_products = []

    for product_id in all_product_ids:
        # if product_id not in purchased_product_ids:
        # Predict the rating for the product for this customer using the trained SVD model
        prediction = svd_model.predict(customer_id, product_id)
        if prediction.est > 1.0:  # You can adjust this threshold based on your needs
            recommended_products.append(product_id)

    # Limit the recommendations to the top 10 products
    top_recommendations = recommended_products[:20]
    
    # Fetch the product names for the recommended products
    recommended_product_names = [product_id for product_id in top_recommendations]

    # Fetch product names for the purchased products
    purchased_product_names = [product_id for product_id in purchased_product_ids]

    return {
        "customer_products": purchased_product_names,  # Return product names instead of IDs
        "recommendations": recommended_product_names,
        "cyclic_purchases": get_cycle_products(customer_id,sales_df)  # Return cyclic purchases for the customer
    }


# Load the saved TF-IDF vectorizer model
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
    
    
def process_and_find_uncommon_products(new_df, row_index):
    
    # print(new_df)
    # Transform 'product_details' using the loaded TF-IDF vectorizer
    product_details_tfidf = tfidf_vectorizer.transform(new_df['product_details'])
    
    # Step 4: Calculate cosine similarity for the specified row
    target_product_detail = new_df.loc[row_index[0], 'product_details']
    # print(target_product_detail)
    target_tfidf = tfidf_vectorizer.transform([target_product_detail])
    similarity_scores = cosine_similarity(target_tfidf, product_details_tfidf).flatten()
    new_df['similarity_score'] = similarity_scores
    
    # Step 5: Sort by similarity score and extract top products
    new_df_sorted = new_df.sort_values(by='similarity_score', ascending=False)
    top_products_details = new_df_sorted['product_details'].head(20)
    # print(new_df_sorted)
    # Step 6: Get product IDs for top similar products
    top_product_ids = new_df[new_df_sorted['product_details'].isin(top_products_details)][['product_id', 'product_details']]
    top_product_ids['product_id_exploded'] = top_product_ids['product_id'].apply(lambda x: x if isinstance(x, list) else [x])
    top_product_ids_exploded = top_product_ids.explode('product_id_exploded')

    product_ids_from_similarity_df = top_product_ids_exploded['product_id_exploded'].unique()
    # Finding uncommon product IDs between exploded product IDs and the second set of product IDs
    uncommon_product_ids = set(top_product_ids_exploded['product_id_exploded']) - set(product_ids_from_similarity_df)

    # Convert the result to a DataFrame for better readability
    uncommon_product_ids_df = pd.DataFrame(list(uncommon_product_ids), columns=['uncommon_product_id'])
    
    # Step 7: Find uncommon product IDs between top two similar products
    product_ids_index_0 = set(top_product_ids.iloc[0]['product_id'])
    product_ids_index_1 = set(top_product_ids.iloc[1]['product_id'])
    uncommon_product_ids = product_ids_index_0 - product_ids_index_1

    # Convert the result to a DataFrame for readability
    uncommon_product_ids_df = pd.DataFrame(list(uncommon_product_ids), columns=['uncommon_product_id'])
    return uncommon_product_ids_df


# <<<<<<<=========day wise recommendation==========>>>>>>>
def get_top_trending_products(df, day_of_week=None):
    """
    Get the top trending products within the last 2 months.
    
    Parameters:
    - df (pd.DataFrame): The data frame containing 'purchased_date_time', 'product_id', 'product_name', and 'quantity'.
    - day_of_week (str): Optional. Specific day of the week to filter (e.g., 'Sunday'). If None, returns top 10 products per day.
    
    Returns:
    - pd.Series: A Series with the top trending product IDs.
    """
    # Convert the 'purchased_date_time' to datetime, handling possible mixed formats
    df['purchased_date_time'] = pd.to_datetime(df['purchased_date_time'], errors='coerce')

    # Drop rows with NaT values that couldn't be parsed
    df = df.dropna(subset=['purchased_date_time'])

    # Set timezone to UTC for consistency
    df['purchased_date_time'] = df['purchased_date_time'].dt.tz_convert('UTC')

    # Define the date range for the past 2 months as a timezone-aware datetime
    two_months_ago = datetime.now(pytz.UTC) - timedelta(days=60)

    # Filter for records in the last 2 months
    filtered_df = df[df['purchased_date_time'] >= two_months_ago]

    # Extract the day of the week from 'purchased_date_time'
    filtered_df['day_of_week'] = filtered_df['purchased_date_time'].dt.day_name()

    # Group by 'day_of_week', 'product_id', and 'product_name', summing the quantities sold
    trending_products = (
        filtered_df.groupby(['day_of_week', 'product_id'])
        .agg({'quantity': 'sum'})
        .reset_index()
    )

    # Sort the products by quantity sold in descending order within each day of the week
    trending_products = trending_products.sort_values(by=['day_of_week', 'quantity'], ascending=[True, False])

    # If a specific day is provided, filter for that day only
    if day_of_week:
        day_trending_products = trending_products[trending_products['day_of_week'] == day_of_week]
        # Get the top 10 products for the specified day
        top_products = day_trending_products.head(20)['product_id']
    else:
        # If no specific day is provided, get the top 10 products for each day
        top_products = trending_products.groupby('day_of_week').head(10)['product_id']

    return top_products.tolist()


def search_recommendation(df, search_term):
    """
    Search recommendation based on a search term.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing 'product_id' and 'product_details'.
    - search_term (str): The search term to find similar products.
    
    Returns:
    - list: A list of top 25 unique product IDs of the most similar products.
    """
    # Transform 'product_details' using the loaded TF-IDF vectorizer
    product_details_tfidf = tfidf_vectorizer.transform([search_term])
    
    # Calculate cosine similarity between search term and product details
    target_tfidf = tfidf_vectorizer.transform(df['product_details'])
    similarity_scores = cosine_similarity(target_tfidf, product_details_tfidf).flatten()
    df['similarity_score'] = similarity_scores
    
    # Sort by similarity score
    df_sorted = df.sort_values(by='similarity_score', ascending=False)
    
    # Extract top 25 unique product IDs
    top_product_ids = df_sorted['product_id'].drop_duplicates().head(25).tolist()
    
    return top_product_ids
