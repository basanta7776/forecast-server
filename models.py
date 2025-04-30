# <<<<<<<<<=====recomedation=========>>>>>>>
from pydantic import BaseModel
from typing import Optional
class CustomerRequest(BaseModel):
    customer_id: str
    
# Define a Pydantic model for the input
class SearchTermRequest(BaseModel):
    search_term: str
    
# Request model for product ID input
class ProductRequest(BaseModel):
    product_id: str
    limit: Optional[int] = 10  # Limit on number of cross-sell products returned


