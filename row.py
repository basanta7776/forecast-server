import pandas as pd
# import pandas as pd

# Load the Excel file
file_path = "products.csv"
df = pd.read_csv(file_path)

# Keep only the first 100,000 rows
df = df.head(40000)

# Save back to a new Excel file
df.to_csv("product_detail.csv", index=False)