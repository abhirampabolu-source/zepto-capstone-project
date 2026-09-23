
import os
import re
import sqlite3
from bs4 import BeautifulSoup
import pandas as pd
import requests

# --- 1. SCRAPING (4 categories to hit >60 books requirement) ---
BASE_URL = "http://books.toscrape.com/"
CATEGORIES = [
    "catalogue/category/books/travel_2/index.html",
    "catalogue/category/books/mystery_3/index.html",
    "catalogue/category/books/historical-fiction_4/index.html",
    "catalogue/category/books/sequential-art_5/index.html",
]

rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
scraped_data = []

for cat_url in CATEGORIES:
    url = BASE_URL + cat_url
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")

    category_name = soup.find("h1").text
    articles = soup.find_all("article", class_="product_pod")

    for art in articles:
        title = art.find("h3").find("a")["title"]
        price_str = art.find("p", class_="price_color").text
        rating_str = art.find("p", class_="star-rating")["class"][1]
        avail_str = art.find("p", class_="instock availability").text.strip()

        scraped_data.append(
            {
                "title": title,
                "price_raw": price_str,
                "rating_raw": rating_str,
                "availability_raw": avail_str,
                "category": category_name,
            }
        )

df = pd.DataFrame(scraped_data)
print(f"Total books scraped: {len(df)}")  # Will print 70

# --- 2. CLEANING & CONVERSION ---
# Currency clean & convert (Fixed Rate: 1 GBP = 105.50 INR)
df["price_gbp"] = df["price_raw"].apply(
    lambda x: float(re.sub(r"[^\d.]", "", x))
)
df["price_inr"] = (df["price_gbp"] * 105.50).round(2)

# Rating mapping (One-Five to 1-5)
df["rating"] = df["rating_raw"].map(rating_map)

# Availability boolean conversion
df["in_stock"] = df["availability_raw"].apply(
    lambda x: 1 if "In stock" in x else 0
)

# Median-imputation handling for numeric fields if any missing
if df["price_gbp"].isnull().any():
    median_val = df["price_gbp"].median()
    df["price_gbp"].fillna(median_val, inplace=True)
    df["price_inr"].fillna(median_val * 105.50, inplace=True)

# --- 3. DATABASE SETUP & POPULATION ---
conn = sqlite3.connect("data_pipeline/zepto_catalog.db")
cursor = conn.cursor()

cursor.executescript(
    """
DROP TABLE IF EXISTS books;
DROP TABLE IF EXISTS categories;

CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE
);

CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
"""
)

# Insert Categories
unique_cats = df["category"].unique()
for cat in unique_cats:
    cursor.execute(
        "INSERT INTO categories (category_name) VALUES (?);", (cat,)
    )

cat_df = pd.read_sql("SELECT * FROM categories;", conn)
df = df.merge(cat_df, left_on="category", right_on="category_name")

# Insert Books
for _, row in df.iterrows():
    cursor.execute(
        """
        INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?);
    """,
        (
            row["title"],
            row["price_gbp"],
            row["price_inr"],
            row["rating"],
            row["in_stock"],
            row["category_id"],
        ),
    )

conn.commit()

# --- 4. EXECUTING 5 REQUIRED SQL QUERIES ---
q1 = "SELECT DISTINCT category_name FROM categories;"
q2 = "SELECT title, price_inr FROM books WHERE price_inr BETWEEN 2000 AND 5000;"
q3 = "SELECT title, price_gbp FROM books WHERE in_stock = 1 ORDER BY price_gbp DESC LIMIT 5;"
q4 = "SELECT title, rating FROM books WHERE rating IN (4, 5);"
q5 = """
SELECT b.title, b.price_gbp, b.price_inr, c.category_name 
FROM books b 
JOIN categories c ON b.category_id = c.category_id;
"""

print("\n--- QUERY 1: DISTINCT ---")
print(pd.read_sql(q1, conn))

print("\n--- QUERY 2: WHERE / BETWEEN ---")
print(pd.read_sql(q2, conn).head())

print("\n--- QUERY 3: ORDER BY / LIMIT ---")
print(pd.read_sql(q3, conn))

print("\n--- QUERY 4: IN ---")
print(pd.read_sql(q4, conn).head())

print("\n--- QUERY 5: SQL JOIN OUTPUT ---")
sql_join_df = pd.read_sql(q5, conn)
print(sql_join_df.head())

# --- 5. PANDAS EQUIVALENT MERGE ---
books_df = pd.read_sql("SELECT * FROM books;", conn)
cats_df = pd.read_sql("SELECT * FROM categories;", conn)

pandas_join_df = pd.merge(
    books_df, cats_df, on="category_id", how="inner"
)[["title", "price_gbp", "price_inr", "category_name"]]

print("\n--- PANDAS MERGE OUTPUT MATCH CHECK ---")
print(pandas_join_df.head())

assert sql_join_df.equals(
    pandas_join_df
), "Outputs between SQL and Pandas do not match!"
print("\nSuccess: SQL and Pandas JOIN outputs match perfectly.")

conn.close()