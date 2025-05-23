from fastapi import FastAPI, File, UploadFile, HTTPException
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*bottleneck.*")
warnings.simplefilter("ignore")
from pydantic import BaseModel
from datetime import timedelta
import pandas as pd
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64, hashlib, time, json, hmac
import os
from prophet import Prophet

# from io import BytesIO
# from typing import Optional

# Secret key for signing the JWT
SECRET_KEY = "your_secret_key_here"

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendRequest(BaseModel):
    user_id: str
    top_n: int = 5

class ProductHistoryRequest(BaseModel):
    user_id: str
    start_date: str



# Encode the JWT
def encode_jwt(payload):
    # Header
    header = {
        "alg": "HS256",
        "typ": "JWT"
    }

    # Base64url encode the header
    header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")

    # Base64url encode the payload
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")

    # Create the signature
    message = f"{header_b64}.{payload_b64}"
    signature = hmac.new(SECRET_KEY.encode(), message.encode(), hashlib.sha256).digest()
    signature_b64 = base64.urlsafe_b64encode(signature).decode().rstrip("=")

    # Return the JWT
    return f"{header_b64}.{payload_b64}.{signature_b64}"

# Decode the JWT (for validation purposes)
def decode_jwt(token):
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
        # Verify the signature
        message = f"{header_b64}.{payload_b64}"
        signature = base64.urlsafe_b64decode(signature_b64 + "==")
        expected_signature = hmac.new(SECRET_KEY.encode(), message.encode(), hashlib.sha256).digest()

        if signature != expected_signature:
            raise ValueError(status_code=401, detail="Invalid token signature")

        # Decode the payload
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "==").decode())
        return payload
    except Exception as e:
        raise ValueError(status_code=400, detail="Invalid token format or payload")


def load_data():
    df = pd.read_csv("data/flipkart_data.csv")
    return df

def preprocess():
    df = pd.read_csv("data/flipkart_com-ecommerce_sample.csv")
    df = df.head(10000)
    df.fillna({'image': 'default_image_url',
               'product_specifications': "",
               'description': "",
               'brand': "Allure Auto",
               'retail_price': df['retail_price'].mean(),
               'discounted_price': df['discounted_price'].mean()}, inplace=True)
    df['overall_rating'] = df['overall_rating'].replace('No rating available', 0)
    df['product_rating'] = df['product_rating'].replace('No rating available', 0)
    df.dropna(inplace=True)
    return df

def find_trending_products(df):
    df['crawl_timestamp'] = pd.to_datetime(df['crawl_timestamp'])
    latest_date = df['crawl_timestamp'].max()
    two_months_ago = latest_date - timedelta(days=15)
    recent_transactions = df[df['crawl_timestamp'] >= two_months_ago]
    # return recent_transactions['product_name']
    return recent_transactions

def fast_forecast_sales(df, days: int = 30):
    if df.empty:
        raise ValueError("Input data is empty.")

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    recent_cutoff = pd.to_datetime("today") - pd.Timedelta(days=365)
    df = df[df['date'] >= recent_cutoff]

    result_list = []

    product_groups = df.groupby(['product_id', 'product_name'])

    for (product_id, product_name), group in product_groups:
        if group.shape[0] < 2:
            continue

        avg_price = group['unit_selling_price'].mean()

        ts = group.groupby('date')['procured_quantity'].sum().reset_index()
        ts = ts.set_index('date').resample('D').sum().fillna(0)
        rolling_avg = ts['procured_quantity'].rolling(window=7, min_periods=1).mean().iloc[-1]

        forecast_dates = pd.date_range(start=pd.to_datetime("today"), periods=days)
        forecast_df = pd.DataFrame({
            'date': forecast_dates.strftime('%Y-%m-%d'),
            'predicted_quantity': [round(rolling_avg, 2)] * days,
            'product_id': product_id,
            'product_name': product_name,
            'predicted_revenue': [round(rolling_avg * avg_price, 2)] * days
        })

        result_list.append(forecast_df)

    if not result_list:
        raise ValueError("No valid forecasts generated.")

    return pd.concat(result_list, ignore_index=True)


class TrendingRequest(BaseModel):
    top_k: int = 10

class SimilarProductRequest(BaseModel):
    search_term: str

class RecommendRequest(BaseModel):
    product_name: str

class ProductHistoryRequest(BaseModel):
    start_date: str  # Format: YYYY-MM-DD
    days: int


class GroupCustomersRequest(BaseModel):
    city_name: str
    top_n: int = 10
    product_name: str

class LoginRequest(BaseModel):
    email: str
    password: str

class CustomerProductRequest(BaseModel):
    customer_id: int  # or use order_id if you prefer


# Login route to generate the JWT
@app.post("/login")
async def login(req: LoginRequest):

    # Simulate user verification (replace with actual DB lookup)
    stored_password = "basanta@123"  # This should be securely hashed and checked
    if req.password != stored_password:
        raise ValueError(status_code=401, detail="Invalid credentials")

    # Create a payload with user info
    payload = {
        "email": req.email,
        "iat": int(time.time()),  # Issued at time (Unix timestamp)
        "exp": int(time.time()) + 3600  # Expiration time (1 hour)
    }

    # Generate JWT
    token = encode_jwt(payload)

    # Return JWT in response
    return JSONResponse(content={"access_token": token, "token_type": "bearer"}, status_code=200)

@app.post("/trending-products", description="Get top trending products based on recent transactions.")
def get_trending_products(req: TrendingRequest):
    df = preprocess()
    # trending_products = find_trending_products(df).to_list()
    # response_data = {"trending_products": trending_products[:req.top_k]}
    # return JSONResponse(content=response_data, status_code=200)
    recent_transaction = find_trending_products(df)['product_name']
    trending_products = recent_transaction.to_list()
    response_data = {"trending_products": trending_products[:req.top_k]}
    return JSONResponse(content=response_data, status_code=200)



@app.post("/similar-products", description="Find similar products based on the search term.")
def get_similar_products(req: SimilarProductRequest):
    df = preprocess()
    similar_products = find_similar_products(req.search_term, df).to_list()
    response_data = {"similar_products": similar_products}
    return JSONResponse(content=response_data, status_code=200)


@app.post("/recommend-products", description="Recommend products similar to the given product name.")
def get_recommended_products(req: RecommendRequest):
    df = preprocess()
    indices = pd.Series(df.index, index=df['product_name']).drop_duplicates()
    tf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tf.fit_transform(df['description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    recommended_products = recommend_similar_products(df, req.product_name, cosine_sim, indices)
    response_data = {"recommended_products": recommended_products}
    return JSONResponse(content=response_data, status_code=200)


@app.post("/products-ordered", description="Get products ordered by a user within a week from the given start date. Format: YYYY-MM-DD")
async def get_products_ordered(req: ProductHistoryRequest):
    data = load_data()
    data['date'] = pd.to_datetime(data['date'])
    start_date = pd.to_datetime(req.start_date)
    end_date = start_date + pd.Timedelta(days=7)
    filtered_df = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
    response_data = {"products": filtered_df['product_name'].unique().tolist()}
    return JSONResponse(content=response_data, status_code=200)


@app.post("/quarterly-forecast", description="Get a quarterly forecast of product sales based on historical data. Format: YYYY-MM-DD")
async def quarter_forecast(req: ProductHistoryRequest):
    data = load_data()
    data['date'] = pd.to_datetime(data['date'])
    start_date = pd.to_datetime(req.start_date)
    end_date = start_date + pd.Timedelta(days=120)
    filtered_df = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
    response_data = {"products": filtered_df['product_name'].unique().tolist()}
    return JSONResponse(content=response_data, status_code=200)
@app.post("/weekly-forecast", description="Get a quarterly forecast of product sales based on historical data. Format: YYYY-MM-DD")
async def quarter_forecast(req: ProductHistoryRequest):
    data = load_data()
    data['date'] = pd.to_datetime(data['date'])
    start_date = pd.to_datetime(req.start_date)
    end_date = start_date + pd.Timedelta(days=7)
    filtered_df = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
    response_data = {"products": filtered_df['product_name'].unique().tolist()}
    return JSONResponse(content=response_data, status_code=200)



@app.post("/forecast-test", description="Get a sales summary for a 7-day period from the given start date.")
async def weekly_sales_report(req: ProductHistoryRequest):
    data = load_data()
    data['date'] = pd.to_datetime(data['date'])

    start_date = pd.to_datetime(req.start_date)
    end_date = start_date + pd.Timedelta(days=req.days)

    # Filter for the week
    filtered_df = data[(data['date'] >= start_date) & (data['date'] < end_date)]

    if filtered_df.empty:
        return JSONResponse(content={"message": "No data found for the given period."}, status_code=404)

    # Calculate metrics
    total_quantity = int(filtered_df['procured_quantity'].sum())
    total_revenue = float((filtered_df['procured_quantity'] * filtered_df['unit_selling_price']).sum())
    avg_selling_price = round(filtered_df['unit_selling_price'].mean(), 2)

    # Group by product
    product_summary = (
        filtered_df.groupby('product_name')
        .agg(total_quantity=('procured_quantity', 'sum'),
             total_revenue=('unit_selling_price', lambda x: (x * filtered_df.loc[x.index, 'procured_quantity']).sum()))
        .reset_index()
        .sort_values(by='total_quantity', ascending=False)
    ).head(25)

    # Convert DataFrame to dict for JSON
    product_summary = product_summary.to_dict(orient="records")

    # Optionally, you can add city or brand-level summaries too
    return JSONResponse(
        content={
            "summary": {
                "total_quantity": total_quantity,
                "total_revenue": total_revenue,
                "avg_selling_price": avg_selling_price,
            },
            "product_summary": product_summary
        },
        status_code=200
    )

@app.post("/monthly-forecast", description="Get a quarterly forecast of product sales based on historical data. Format: YYYY-MM-DD")
async def quarter_forecast(req: ProductHistoryRequest):
    data = load_data()
    data['date'] = pd.to_datetime(data['date'])
    start_date = pd.to_datetime(req.start_date)
    end_date = start_date + pd.Timedelta(days=30)
    filtered_df = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
    response_data = {"products": filtered_df['product_name'].unique().tolist()}
    return JSONResponse(content=response_data, status_code=200)
@app.post("/yearly-forecast", description="Get a quarterly forecast of product sales based on historical data. Format: YYYY-MM-DD")
async def quarter_forecast(req: ProductHistoryRequest):
    data = load_data()
    data['date'] = pd.to_datetime(data['date'])
    start_date = pd.to_datetime(req.start_date)
    end_date = start_date + pd.Timedelta(days=365)
    filtered_df = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
    response_data = {"products": filtered_df['product_name'].unique().tolist()}
    return JSONResponse(content=response_data, status_code=200)

@app.post("/daily-forecast", description="Get a daily forecast of product sales based on historical data. Format: YYYY-MM-DD")
async def day_forecast(req: ProductHistoryRequest):
    data = load_data()
    data['date'] = pd.to_datetime(data['date'])
    start_date = pd.to_datetime(req.start_date)
    end_date = start_date + pd.Timedelta(days=1)
    filtered_df = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
    response_data = {"products": filtered_df['product_name'].unique().tolist()}
    print(response_data)
    return JSONResponse(content=response_data, status_code=200)


class ProductHistoryRequest(BaseModel):
    days: int = 30

@app.post("/forecast", description="Fast forecast of product sales.")
async def forecast_endpoint(req: ProductHistoryRequest):
    try:
        data = load_data()
        forecast_df = fast_forecast_sales(data, days=req.days)
        forecast_df = forecast_df.head(30)
        total_revenue = round(forecast_df['predicted_revenue'].sum(), 2)

        return JSONResponse(
            content={
                "forecast": forecast_df.to_dict(orient='records'),
                "total_predicted_revenue": total_revenue
            },
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# @app.post("/group-customers", description="Get a list of user IDs belonging to a specific city.")
# def get_users_by_city(req: GroupCustomersRequest):
#     df = load_data()
#     length = len(df[df['city_name'].str.lower() == req.city_name.lower()]['order_id'].tolist())
#     if 'order_id' not in df.columns or 'city_name' not in df.columns:
#         raise ValueError("DataFrame must contain 'order_id' and 'cityname' columns")

#     # if length < req.top_n:
#     # else:
#         # filtered_users = df[df['city_name'].str.lower() == req.city_name.lower()]['order_id'].tolist()[:req.top_n]
#     filtered_users = df[df['city_name'].str.lower() == req.city_name.lower()]['order_id'].tolist()

#     city_data = df[df['city_name'].str.lower() == req.city_name.lower()]
#     # Average order value
#     # avg_order_value = city_data['order_id'].mean()

#     total_orders = city_data['order_id'].count()
#     total_order_full = df["order_id"].count()

#     avg_orders = total_order_full / total_orders
#     print(avg_orders)

#     response_data = {
#         "list":filtered_users,
#         "average_order_value": round(avg_orders, 2),
#     }
#     return JSONResponse(content=response_data, status_code=200)

import logging

logging.basicConfig(level=logging.DEBUG)


# @app.post("/group-customers", description="Get a list of user details belonging to a specific city.")
# def get_users_by_city(req: GroupCustomersRequest):
#     df = load_data()

#     # Check if 'order_id' and 'city_name' are present in the dataframe
#     if 'order_id' not in df.columns or 'city_name' not in df.columns:
#         raise ValueError("DataFrame must contain 'order_id' and 'city_name' columns")

#     # Filter data for the specific city
#     city_data = df[df['city_name'].str.lower() == req.city_name.lower()]

#     # Get the user data (not just 'order_id')
#     user_details = city_data[['order_id', 'product_id', 'product_name', 'quantity', 'product_type',
#                               'brand', 'manufacturer_name', 'date', 'city_name', 'dim_customer_key',
#                               'procured_quantity', 'unit_selling_price', 'total_discount_amount']].head(20)

#     response_data = {
#         "user_details": user_details.to_dict(orient='records'),
#     }

#     logging.debug("Returning response data: %s", response_data)
#     return JSONResponse(content=response_data, status_code=200)
@app.post("/group-customers", description="Get user details for a specific product in a city.")
def get_users_by_product_in_city(req: GroupCustomersRequest):
    df = load_data()

    if 'order_id' not in df.columns or 'city_name' not in df.columns:
        raise ValueError("DataFrame must contain 'order_id' and 'city_name' columns")

    # Filter by city and product
    city_product_data = df[
        (df['city_name'].str.lower() == req.city_name.lower()) &
        (df['product_name'].str.lower() == req.product_name.lower())
    ]

    if city_product_data.empty:
        return JSONResponse(content={"user_details": []}, status_code=200)

    # Limit results
    user_details = city_product_data[[
        'order_id', 'product_id', 'product_name', 'quantity', 'product_type',
        'brand', 'manufacturer_name', 'date', 'city_name', 'dim_customer_key',
        'procured_quantity', 'unit_selling_price', 'total_discount_amount'
    ]].head(req.top_n)

    response_data = {
        "user_details": user_details.to_dict(orient='records')
    }

    logging.debug("Returning response data: %s", response_data)
    return JSONResponse(content=response_data, status_code=200)


@app.get("/sales_report", description="Get an average sales amount based on the order placed.")
def sales_report():
    df = load_data()

    if 'unit_selling_price' not in df.columns or 'total_discount_amount' not in df.columns or 'order_id' not in df.columns:
        raise ValueError("Required columns missing in data.")

    # # Total net sale amount = price - discount for each row
    df['net_sale'] = df['unit_selling_price'] - df['total_discount_amount']
    # # Total sales amount
    total_sales =  df['net_sale'].sum()

    response_data = {"total_sales": float(total_sales)}
    return JSONResponse(content=response_data, status_code=200)


# @app.get("/low_stock", description="Get a list of user IDs belonging to a specific city.")
# def low_stock():
#     df = preprocess()
#     trending_products = find_trending_products(df)
#     print(trending_products.columns)
#     low_df = trending_products.sort_values(by='retail_price', ascending=True)

#       # Select top 15 with product_name and uniq_id
#     low_stock = low_df[['uniq_id', 'product_name']].head(15).to_dict(orient='records')

#     response_data = {"low_stocks": low_stock}
#     return JSONResponse(content=response_data, status_code=200)

import re

def extract_numeric_quantity(qty_str):
    """Extracts the numeric part from '1 kg' or '120 units'."""
    match = re.match(r"(\d+(?:\.\d+)?)", qty_str)
    return float(match.group(1)) if match else 0

@app.get("/low_stock", description="Get low stock items with product info.")
def low_stock():
    df = load_data()

    # Extract numeric value from quantity string
    df['numeric_quantity'] = df['quantity'].apply(extract_numeric_quantity)

    # Group by product and compute totals
    summary = (
        df.groupby(['product_id', 'product_name', 'brand', 'manufacturer_name'], as_index=False)
        .agg({
            'numeric_quantity': 'sum',
            'procured_quantity': 'sum',
            'unit_selling_price': 'min'
        })
        .rename(columns={'numeric_quantity': 'total_quantity'})
    )

    # Filter for low stock: both procured and total quantity low
    low_stock_df = summary[
        (summary['procured_quantity'] <= 5) & (summary['total_quantity'] <= 20)
    ].sort_values(by='procured_quantity', ascending=True).head(15)

    # Convert to dict for response
    low_stock_list = low_stock_df.to_dict(orient='records')
    response_data = {"low_stocks": low_stock_list}

    return JSONResponse(content=response_data, status_code=200)


from fastapi import UploadFile, File
from fastapi.responses import JSONResponse

UPLOAD_DIR = "./data"

@app.post("/upload-csv", description="Upload a CSV file and save to /data/")
async def upload_csv(file: UploadFile = File(...)):
    # Check if file is a CSV
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure /data directory exists
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # Write the uploaded file to disk
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        return JSONResponse(content={"message": f"File '{file.filename}' uploaded successfully."}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

@app.post("/upload-file", description="Upload a file and save it to ./data/")
async def upload_file(file: UploadFile = File(...)):
    # Ensure the /data directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Create the full path for the file to be saved
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # Write the uploaded file to disk
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Return a success message with the file name
        return JSONResponse(content={"message": f"File '{file.filename}' uploaded successfully."}, status_code=200)

    except Exception as e:
        # If an error occurs, raise an HTTPException
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")


# app = FastAPI()

# In-memory storage for the uploaded CSV
# uploaded_data = None


# @app.get("/generate-report", description="Generate a report based on the uploaded CSV")
# async def generate_report():
#     global uploaded_data

#     if uploaded_data is None:
#         raise HTTPException(status_code=400, detail="No CSV file uploaded.")

#     # Example report: top 5 products by frequency
#     report = uploaded_data['product_name'].value_counts().head(5).to_dict()
#     return JSONResponse(content={"report": report}, status_code=200)

###########



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000,reload=True)


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


#     # command to run the app in terminal :
#     # uvicorn main:app --host 0.0.0.0 --port 8000 --reload

