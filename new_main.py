from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
# from surprise import SVD, Dataset, Reader,dump
# from surprise.model_selection import train_test_split
# from surprise import accuracy

# from surprise.model_selection import train_test_split
from surprise import dump
# from surprise import accuracy


app = FastAPI()
# Sample Data for Demo purposes
# df = pd.read_csv("/path/to/your/flipkart_data.csv")
# # Recommender Model
# reader = Reader(rating_scale=(1, 5))
# data = Dataset.load_from_df(df[['uniq_id', 'description', 'overall_rating']], reader)
# trainset, testset = train_test_split(data, test_size=0.2)
# model = SVD()
# model.fit(trainset)
class RecommendRequest(BaseModel):
    user_id: str
    top_n: int = 5

class ProductHistoryRequest(BaseModel):
    user_id: str
    start_date: str

# Endpoint for product recommendations

def load_data():
    df= pd.read_csv("Sales_forecast/data/flipkart_data.csv")
    return df
def load_svd_model():
    model_file = 'Sales_forecast/model.pkl'
    _, svd_model = dump.load(model_file)
    return svd_model

  # Load the trained SVD model when the app starts




@app.post("/recommend-products/")
async def recommend_products(req: RecommendRequest):
    df =load_data()
    model = load_svd_model()
    unique_products = df[['product_id', 'product_name']].drop_duplicates()
    predictions = [model.predict(req.user_id, pid) for pid in unique_products['product_id']]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_product_ids = [pred.iid for pred in predictions[:req.top_n]]
    recommended_products = unique_products[unique_products['product_id'].isin(top_product_ids)]
    return recommended_products[['product_id', 'product_name']].to_dict(orient="records")

# Endpoint for getting purchased products in the last 7 days
@app.post("/products-ordered/")
async def get_products_ordered(req: ProductHistoryRequest):
    data = load_data()
    # data = pd.read_csv("/path/to/your/data.csv")
    data['date'] = pd.to_datetime(data['date'])
    start_date = pd.to_datetime(req.start_date)
    end_date = start_date + pd.Timedelta(days=7)
    filtered_df = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
    filtered_product_names = filtered_df['product_name'].unique().tolist()
    return {"products": filtered_product_names}

# Endpoint for quarterly product forecast
@app.post("/quarterly-forecast/")
async def quarter_forecast(req: ProductHistoryRequest):
    data = load_data()
    data['date'] = pd.to_datetime(data['date'])
    start_date = pd.to_datetime(req.start_date)
    end_date = start_date + pd.Timedelta(days=120)
    filtered_df = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
    filtered_product_names = filtered_df['product_name'].unique().tolist()
    return {"products": filtered_product_names}

# Endpoint for daily product forecast
@app.post("/daily-forecast/")
async def day_forecast(req: ProductHistoryRequest):
    # data = pd.read_csv("/path/to/your/data.csv")
    data = load_data()
    data['date'] = pd.to_datetime(data['date'])
    start_date = pd.to_datetime(req.start_date)
    end_date = start_date + pd.Timedelta(days=1)
    filtered_df = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
    filtered_product_names = filtered_df['product_name'].unique().tolist()
    return {"products": filtered_product_names}

# Endpoint to forecast product rating for user and item
@app.get("/forecast-rating/")
async def forecast_rating(user_id: str, item_id: str):
    model  =load_svd_model()
    predicted_rating = model.predict(user_id, item_id)
    return {"predicted_rating": predicted_rating.est}

# Endpoint for market analysis (specific date)
# @app.post("/market-analysis/")
# async def market_analysis(start_date: str):
#     ecom = pd.read_csv("/path/to/your/flipkart_data.csv")
#     f_sales = pd.read_csv("/path/to/your/flipkart_sales_data.csv")
#     merged_data = pd.merge(ecom, f_sales, on='product_name')
#     merged_data['date'] = pd.to_datetime(merged_data['date'])
#     start_date = pd.to_datetime(start_date)
#     end_date = start_date + pd.Timedelta(days=1)
#     filtered_df = merged_data[(merged_data['date'] >= start_date) & (merged_data['date'] <= end_date)]
#     total_transaction = filtered_df['unit_selling_price'].sum()
#     return {"total_transaction": total_transaction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
