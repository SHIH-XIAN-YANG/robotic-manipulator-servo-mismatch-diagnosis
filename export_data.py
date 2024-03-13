#%%

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import json
import numpy as np
import pymysql
import time
import csv


# connect to databse
connction = pymysql.connect(
    host = "localhost",
    user="root",
    port=3306,
    password="Sam512011",
    database="bw_mismatch_db"
)

cursor = connction.cursor()


"""
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|  id |  Gain | BW | C_error | min_bandwidth | t_errX | t_errY | t_errZ | tracking_err_pitch | tracking_err_roll | tracking_err_yaw | contour_err_img_path | ori_contour_err_img_path|
|-----|-------|----|---------|---------------|--------|--------|--------|--------------------|-------------------|------------------|----------------------|-------------------------|
...

"""
# sql = "SELECT Gain FROM bw_mismatch_data;"
# cursor.execute(sql)
# gain = np.array([])
# data = cursor.fetchall()
# for idx, row in enumerate(data):
#     # print(json.loads(row[0]))
#     gain = np.append(gain, json.loads(row[0]),axis=0)
# gain = gain.reshape((-1,len(json.loads(row[0]))))


# sql = "SELECT Bandwidth FROM bw_mismatch_data;"
# cursor.execute(sql)
# bandwidth = np.array([])
# data = cursor.fetchall()
# for idx, row in enumerate(data):
#     # print(json.loads(row[0]))
#     bandwidth = np.append(bandwidth, json.loads(row[0]))
# bandwidth = bandwidth.reshape((-1,len(json.loads(row[0]))))

# print(bandwidth)
sql = "SELECT min_bandwidth FROM bw_mismatch_data;"
cursor.execute(sql)
min_bandwidth = np.array([])
data = cursor.fetchall()
for idx, row in enumerate(data):
    # print(row[0])
    arr = [0]*6
    arr[row[0]] = 1
    min_bandwidth = np.append(min_bandwidth,arr)
min_bandwidth = min_bandwidth.reshape((-1, len(arr)))


sql = "SELECT contour_err FROM bw_mismatch_data;"
cursor.execute(sql)
contour_err = np.array([])
data = cursor.fetchone()
for idx, row in enumerate(data):
    # print(len(json.loads(row)))

    contour_err = np.append(contour_err, json.loads(row))
# print(row)
contour_err = contour_err.reshape((-1, len(json.loads(row))))

sql = "SELECT ori_contour_err FROM bw_mismatch_data;"
cursor.execute(sql)
ori_contour_err = np.array([])
data = cursor.fetchone()
for idx, row in enumerate(data):
    # print(len(json.loads(row[0])))
    ori_contour_err = np.append(ori_contour_err, json.loads(row))
# print(row)
ori_contour_err = ori_contour_err.reshape((-1, len(json.loads(row))))
print(ori_contour_err)


# Initialize a dictionary to store the counts
distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
print(min_bandwidth)
# Count the distribution of numbers
for num in (min_bandwidth):
    
    distribution[np.argmax(num)] += 1



# Print the distribution
for key, value in distribution.items():
    print(f"Number {key}: Count {value}")
# Extract keys and values for plotting
keys = list(distribution.keys())
values = list(distribution.values())

# Plot the distribution
plt.figure()
plt.bar(keys, values, color='skyblue')
plt.xlabel('joint')
plt.ylabel('Count')
plt.title('Distribution of Numbers')
plt.xticks(keys)  # Set x-ticks to show all numbers from 0 to 5
plt.grid(True)
plt.show()
fig, axs = plt.subplots(2, 1, figsize=(8, 6))
axs[0].plot(contour_err[0], color='blue')
axs[0].set_title('contour_err', fontsize=18)
axs[0].set_xlabel('X-axis')
axs[0].set_ylabel('contour_err(mm)', fontsize=18)
axs[0].grid()
axs[1].plot(ori_contour_err[0], color='red')
axs[1].set_title('ori_contour_err', fontsize=18)
axs[1].set_xlabel('X-axis')
axs[1].set_ylabel('ori(deg)', fontsize=18)
axs[1].grid()
plt.show()
# File path to save the CSV file
file_path = f'min_bandwidth.csv'

# Writing to CSV file
with open(file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(min_bandwidth)

print(f"Data exported to {file_path}")



# Load data from CSV file using numpy
data = np.genfromtxt(file_path, delimiter=',', dtype=str)

min_bandwidth = []

for row in data:
    min_bandwidth.append(np.argmax(row, axis=0))

min_bandwidth = np.array(min_bandwidth)
print(min_bandwidth)
