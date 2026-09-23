# Module 1: Data Pipeline

## Setup & Execution
1. Install dependencies: `pip install -r requirements.txt`
2. Run data pipeline script: `python data_pipeline/main.py`

## Design & Cleaning Decisions
- **Data Scope**: Scraped 71 products across 4 categories (`Travel`, `Mystery`, `Historical Fiction`, `Sequential Art`) to exceed the >= 60 books requirement.
- **Fixed Currency Conversion**: Converted GBP to INR using the exact required project constant `1 GBP = 105.50 INR`.
- **Data Imputation**: Configured numeric imputation using median values for any missing or unparseable numeric fields to ensure execution resilience.
- **Normalized Database Schema**: Designed a 2-table normalized relational SQLite store (`categories` and `books`) linked via a Foreign Key (`category_id`).
- **Query & Merge Verification**: Executed 5 SQL queries covering `SELECT`, `WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `BETWEEN`, `IN`, and `JOIN`. Verified that `pandas.merge()` on raw dataframes matches `pd.read_sql()` JOIN output identically.