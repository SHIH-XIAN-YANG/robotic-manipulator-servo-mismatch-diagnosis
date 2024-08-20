import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float
from sqlalchemy.dialects.mysql import VARCHAR
import json
# Define the MySQL connection parameters
host = "localhost",
user="root",
port=3306,
password="Sam512011",
database="bw_mismatch_db"

# Create the connection engine
engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')

# Read the CSV file into a DataFrame
csv_file_path = 'D:\\mismatch_joints_dataset.csv'
df = pd.read_csv(csv_file_path)

# Get the column names and types from the DataFrame
columns = df.columns
column_types = df.dtypes

# Define a function to map pandas dtypes to SQLAlchemy types
def map_dtype(dtype, column):
    if pd.api.types.is_integer_dtype(dtype):
        return Integer
    elif pd.api.types.is_float_dtype(dtype):
        return Float
    elif pd.api.types.is_object_dtype(dtype):
        sample_value = df[column].dropna().iloc[0]
        if isinstance(sample_value, str):
            try:
                json.loads(sample_value)
                return JSON
            except (ValueError, TypeError):
                return VARCHAR(255)
        else:
            return VARCHAR(255)
    else:
        return String

# Define the table name
table_name = 'mismatch_joints_dataset'

# Reflect existing tables
metadata = MetaData()
metadata.reflect(bind=engine)

# Check if the table already exists
if table_name not in metadata.tables:
    # Define the table structure
    table = Table(
        table_name,
        metadata,
        *(Column(col, map_dtype(dtype)) for col, dtype in zip(columns, column_types))
    )

    # Create the table in the database
    metadata.create_all(engine)
else:
    table = metadata.tables[table_name]

# Import the data into the table
df.to_sql(table_name, engine, if_exists='append', index=False)

print("Data imported successfully!")