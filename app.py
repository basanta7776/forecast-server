from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel
from utils import answer_question, expert_system, parse_input_string, extract_text_from_image, extract_texts_from_pdf, split_and_translate
import json
from models import contextReply, analysisReport, QueryInput,  QueryInputBetter, TextExtractionResponse, CustomerRequest, SearchTermRequest
from utilss import recommend_products_for_customer, process_and_find_uncommon_products, get_top_trending_products, search_recommendation
import os
import uuid
import pandas as pd


# <<<<<<<<<<<=================== recomendation ===================>>>>>>>>>>>>>>>
from collections import defaultdict
app = FastAPI()
@app.post("/recommendation/predict/")
async def predict_products(request: CustomerRequest):
    """
    
    Predicts the products that a customer is likely to purchase based on their past interactions.
    
    Args:
    - request (CustomerRequest): The request object containing the customer ID.
    
    
    Returns:
    - customer_id (str): The unique identifier of the customer.
    - purchased_products (list): The products that the customer has already purchased.
    - recommended_products (list): The products recommended for the customer.
    - cyclic_purchases (dict): The cyclic purchases made by the customer.
    """
    
    customer_id = request.customer_id

    # Get recommendations for the customer
    result = recommend_products_for_customer(customer_id)

    if "message" in result and result["message"] == "Customer ID not found in the dataset.":
        raise HTTPException(status_code=404, detail="Customer ID not found")

    return {
        "customer_id": customer_id,
        "purchased_products": result["customer_products"],
        "recommended_products": result["recommendations"],
        "cyclic_purchases": result["cyclic_purchases"]  # Return cyclic purchases if available
    }


# Load the CSV file from the local data folder
df = pd.read_csv('data/df.csv')
filtered_df = None
product_associations = defaultdict(set)

@app.post("/recommendation/find-uncommon-products/")
async def find_uncommon_products(customer_id: str):
    """
    Find uncommon products that a customer has purchased compared to other customers.
    
    Args:
    - customer_id (str): The unique identifier of the customer.
    
    Returns:
    - recomending_uncommon_product_ids (list): The list of uncommon product IDs recommended for the customer.
    """
    
    
    # Step 1: Group by 'customer_id' and aggregate required columns
    new_df = df.groupby('customer_id').agg({
        'product_id': list,
        'product_name': list,
        'product_description': list
    }).reset_index()
    
    # Step 2: Combine 'product_name' and 'product_description' into 'product_details'
    def merge_name_description(names, descriptions):
        return [f"{name}: {desc}" if desc is not None else name for name, desc in zip(names, descriptions)]
    
    new_df['product_details'] = new_df.apply(lambda row: merge_name_description(row['product_name'], row['product_description']), axis=1)
    new_df = new_df.drop(columns=['product_name', 'product_description'])
    new_df['product_details'] = new_df['product_details'].apply(lambda x: ' '.join(x))

    row_index = new_df.index[new_df['customer_id'] == customer_id].tolist()
    print(row_index)
    
    if row_index == []:
        raise HTTPException(status_code=404, detail="Customer ID not found")
    
    # Process and find uncommon products
    uncommon_product_ids_df = process_and_find_uncommon_products(new_df, row_index=row_index)

    # Convert the result to JSON format
    return {"recomending_uncommon_product_ids": uncommon_product_ids_df['uncommon_product_id'].tolist()}


@app.get("/recommendation/top-trending-products/")
async def get_top_trending_products_api(day_of_week: str = None):
    """
    Get the top trending products within the last 2 months.
    
    Args:
    - day_of_week (str): The day of the week for which to get the top trending products (Monday).
    
    Returns:
    - top_trending_products (list): The list of top trending products.
    """
    
    # Get the top trending products
    top_trending_products = get_top_trending_products(df, day_of_week=day_of_week)
    
    return {"top_trending_products": top_trending_products}




# recommedndation based on search products name
# Recommendation based on search product name# Recommendation based on search product name
@app.post("/recommendation/search-recommendation/")
async def find_search_products(request: SearchTermRequest):
    """
    Find top 25 similar products based on a search term.
    
    Args:
    - search_term (str): The search term to find similar products.
    
    Returns:
    - recomending_product_ids_search (list): The list of top 25 unique product IDs based on similarity.
    """
    
    search_term = request.search_term
    
    # Prepare the product details by merging 'product_name' and 'product_description'
    df['product_details'] = df.apply(lambda row: f"{row['product_name']}: {row['product_description']}" if row['product_description'] is not None else row['product_name'], axis=1)
    
    # Process and find search products
    search_product_ids = search_recommendation(df, search_term)

    # Convert the result to JSON format
    return {"recomending_product_ids_search": search_product_ids}


# Load the CSV file and preprocess
def load_data():
    global base_df, filtered_df, product_associations
    
    base_df = df
    base_df['purchased_date_time'] = pd.to_datetime(base_df['purchased_date_time'], errors='coerce', format='ISO8601')
    
    # Filter data between 2023 and 2024
    start_date = '2023-01-01'
    end_date = '2024-12-31'
    filtered_df = base_df[(base_df['purchased_date_time'] >= start_date) & (base_df['purchased_date_time'] <= end_date)]
    
    # Build product associations
    for products in filtered_df.groupby('customer_id')['product_id'].apply(list):
        for product in products:
            product_associations[product].update(p for p in products if p != product)

# Define a function to get associated products
def get_associated_products(product_id: str):
    return list(product_associations.get(product_id, []))

# Load data initially
load_data()

from models import ProductRequest

@app.post("/recommendation/cross-sell")
def get_cross_sell_products(request: ProductRequest):
    product_id = request.product_id
    limit = request.limit

    # Get the actual product name
    # actual_product_name = get_product_names([product_id])
    # if not actual_product_name:
    #     raise HTTPException(status_code=404, detail="Product not found")

    # Get associated product IDs
    associated_products = get_associated_products(product_id)
    if not associated_products:
        return {"message": "No associated products found for this product ID"}

    # Get cross-sell product names
    # cross_sell_product_names = get_product_names(associated_products[:limit])

    return {
        "Actual Product": product_id,
        "Cross-sell Product Names": associated_products[:limit]
    }
