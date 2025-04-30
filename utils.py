import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import re
from datetime import datetime, timedelta

# df = pd.read_csv("E:/Projects/College Projetcs/Sales_forecast/flipkart_com-ecommerce_sample.csv")
def preprocess(df):
    df['image'].fillna('["http://img5a.flixcart.com/image/short/u/4/a/altht-3p-21-alisha-38-original-imaeh2d5vm5zbtgg.jpeg", "http://img5a.flixcart.com/image/short/p/j/z/altght4p-26-alisha-38-original-imaeh2d5kbufss6n.jpeg", "http://img5a.flixcart.com/image/short/p/j/z/altght4p-26-alisha-38-original-imaeh2d5npdybzyt.jpeg", "http://img5a.flixcart.com/image/short/z/j/7/altght-7-alisha-38-original-imaeh2d5jsz2ghd6.jpeg"]',inplace = True)
    df['product_specifications'].fillna("",inplace =True)
    df['description'].fillna("",inplace =True)
    df["brand"].fillna("Allure Auto", inplace =True)
    df['retail_price'].fillna(df['retail_price'].mean(),inplace =True)
    df['discounted_price'].fillna(df['discounted_price'].mean(),inplace =True)
    df =  df.dropna()
    #  Data processing
    df['overall_rating'] = df['overall_rating'].replace('No rating available', 0)
    df['product_rating'] = df['product_rating'].replace('No rating available', 0)
    return df

# # taking tfidf vector of description
# tf = TfidfVectorizer(stop_words='english')
# documents = df['description']
# tfidf_matrix = tf.fit_transform(documents)
# cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# sim_df = pd.DataFrame(cosine_sim)
# indices = pd.Series(df.index, index = df['product_name']).drop_duplicates()

def recommend_similar_products(df,product_name, cosine_sim,indices ):
  idx = indices[product_name]
  linear_scores = list(enumerate(cosine_sim[idx]))
  linear_scores = sorted(linear_scores, key = lambda x: x[1], reverse = True)
  linear_scores = linear_scores[0:11]
  product_indices = [x[0] for x in linear_scores]
  scores = [x[1] for x in linear_scores]
  lists = list(df['product_name'].iloc[product_indices])
  return lists

# product_list = pd.Series(df['product_name'], index = df.index).drop_duplicates()
# random_number = random.randint(0, 100)
# random_product_name = product_list[random_number]
# #random_product_name = "Sicons All Purpose Arnica Dog Shampoo"cls

# random_product_name

# Example of recommendation 
# print("Below are the recommendations for the product -",  random_product_name, "\n")
# print(recommend_similar_products(random_product_name))

def find_similar_products(search_term, data_frame, top_n=25):
    descriptions = data_frame['description'].fillna('').str.lower()
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    search_vector = vectorizer.transform([search_term.lower()])
    similarity_scores = cosine_similarity(search_vector, tfidf_matrix).flatten()
    data_frame['similarity_score'] = similarity_scores
    # df = df.sort_values(by='similarity_score', ascending=False)
    similar_products = data_frame.drop_duplicates(subset='similarity_score',keep='first').sort_values(by='similarity_score',ascending = False).head(top_n).reset_index(drop=True)
    # similar_products = df.sort_values(by='similarity_score', ascending=False).unique().head(top_n).reset_index(drop=True)
    return similar_products['product_name']

# example
# similar_products = find_similar_products('shoes', df)
# similar_products.to_list()

def process_row(text):
  pattern = r'(\d+\.?\d*)\s*(mg|ml|l|ML|MG|L|Mg|Tablet)(?:\b|$)'
  match = re.search(pattern, text)

  if match:
      value, unit = match.groups()
      # Standardize unit to lowercase
      unit = unit.lower()
      # Remove the quantity part from product name
      name = re.sub(pattern, '', text).strip()
      return pd.Series([name, float(value), unit])
  else:
      # If no quantity found, return original text as name
      return pd.Series([text, None, ''])



def find_trending_products(df):
  df['crawl_timestamp'] = pd.to_datetime(df['crawl_timestamp'])
  latest_date = df['crawl_timestamp'].max()
  two_months_ago = latest_date - timedelta(days=62)
  recent_transactions = df[df['crawl_timestamp'] >= two_months_ago]
  trending_products = recent_transactions['product_name'].value_counts().sort_values(ascending=False)
  trending_products
  return trending_products

# testing the trend product.
# trending_products = find_trending_products(df)
# trending_products
# print(df)