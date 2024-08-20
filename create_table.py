import pymysql
import pandas as pd
from tqdm import tqdm
from tqdm.contrib import tzip

# MySQL connection setup
connection = pymysql.connect(
    host="localhost",
    user="root",
    port=3306,
    password="Sam512011",
    database="bw_mismatch_db"
)

def create_table_from_csv(file_path, connection):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Infer table name from file name (without extension)
    table_name = file_path.split('/')[-1].split('.')[0]
    
    # Create a cursor object
    cursor = connection.cursor()
    
    # Generate SQL for creating table
    columns = []
    for col_name, col_type in tzip(df.columns, df.dtypes):
        if col_type == 'int64':
            sql_type = 'INT'
        else:
            sql_type = 'JSON'
        columns.append(f"`{col_name}` {sql_type}")
    
    create_table_sql = f"CREATE TABLE IF NOT EXISTS `{table_name}` ({', '.join(columns)});"
    
    # Execute the SQL command to create the table
    cursor.execute(create_table_sql)
    
    # Prepare the SQL statement for inserting the data
    placeholders = ', '.join(['%s'] * len(df.columns))
    insert_sql = f"INSERT INTO `{table_name}` VALUES ({placeholders})"
    
    # Insert data into the table row by row
    for row in tqdm(df.itertuples(index=False, name=None)):
        cursor.execute(insert_sql, row)
    
    # Commit the transaction
    connection.commit()

    # Close the cursor
    cursor.close()


# Load the CSV file into a DataFrame
file_path = 'D://mismatch_joints_dataset.csv'  # Replace with your actual CSV file path
create_table_from_csv(file_path, connection)

# Close the connection
connection.close()

df = pd.read_csv(file_path)

# Print the field names (column headers)
print("Field Names:")
print(df.columns.tolist())

print(df.iloc[0][1])
# # Print the first row of data
# print("\nFirst Row of Data:")
# print(df.iloc[0].tolist())