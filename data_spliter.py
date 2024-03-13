# connect to databse
import json
import pymysql
import os
import shutil
import glob

def list_image_files(directory):
    # Define image file extensions you want to search for
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp']
    image_files = []
    
    # Enumerate through each extension and search for matching files
    for ext in image_extensions:
        # Use glob to match files with the current extension
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    
    return image_files

connction = pymysql.connect(
    host = "localhost",
    user="root",
    port=3306,
    password="Sam512011",
    database="bw_mismatch_db"
)

cursor = connction.cursor()


"""
+-----------------------------------------------------------------------+
|  id |  Gain | BW | C_error | max_bandwidth | t_errX | t_errY | t_errZ |
|-----|-------|----|---------|---------------|--------|--------|--------|
...

"""
sql = "SELECT ori_contour_err_img_path FROM bw_mismatch_data"
cursor.execute(sql)
data = cursor.fetchone()[0]

print(os.path.dirname(data))
dir = os.path.dirname(data)

for i in range(6):
    if not os.path.exists(dir + f'\\{i+1}\\'):
        os.makedirs(dir + f'\\{i+1}\\')

image_files = list_image_files(dir)


for image_file in image_files:
    id = os.path.splitext(os.path.basename(image_file))[0]
    sql = f"SELECT max_bandwidth FROM bw_mismatch_data WHERE id = {id}"
    cursor.execute(sql)
    max_bw = cursor.fetchone()[0]
    # print(id, max_bw)
    shutil.copy(image_file, dir + f'\\{max_bw}\\')