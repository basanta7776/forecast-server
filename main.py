from fastapi import FastAPI
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
import base64
import hashlib
import hmac
import json
import time

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

class LoginRequest(BaseModel):
    email: str
    password: str


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
    end_date = start_date + pd.Timedelta(days=360)
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


@app.post("/group-costumers", description="Get a list of user IDs belonging to a specific city.")
def get_users_by_city(req: GroupCustomersRequest):
    df = load_data()
    length = len(df[df['city_name'].str.lower() == req.city_name.lower()]['order_id'].tolist())
    if 'order_id' not in df.columns or 'city_name' not in df.columns:
        raise ValueError("DataFrame must contain 'order_id' and 'cityname' columns")

    # if length < req.top_n:
    # else:
        # filtered_users = df[df['city_name'].str.lower() == req.city_name.lower()]['order_id'].tolist()[:req.top_n]
    filtered_users = df[df['city_name'].str.lower() == req.city_name.lower()]['order_id'].tolist()

    city_data = df[df['city_name'].str.lower() == req.city_name.lower()]
    # Average order value
    # avg_order_value = city_data['order_id'].mean()

    total_orders = city_data['order_id'].count()
    total_order_full = df["order_id"].count()

    avg_orders = total_order_full / total_orders
    print(avg_orders)

    response_data = {
        "list":filtered_users,
        "average_order_value": round(avg_orders, 2),
    }
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


@app.get("/low_stock", description="Get a list of user IDs belonging to a specific city.")
def low_stock():
    df = preprocess()
    trending_products = find_trending_products(df)
    print(trending_products.columns)
    low_df = trending_products.sort_values(by='retail_price', ascending=True)

      # Select top 15 with product_name and uniq_id
    low_stock = low_df[['uniq_id', 'product_name']].head(15).to_dict(orient='records')

    response_data = {"low_stocks": low_stock}
    return JSONResponse(content=response_data, status_code=200)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


#     # command to run the app in terminal :
#     # uvicorn main:app --host 0.0.0.0 --port 8000 --reload

