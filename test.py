#%%
import json
import numpy as np
import pymysql
import time

class TestClass():
    def __init__(self):
        print("import success")
    
    



# connect to databse
connction = pymysql.connect(
    host = "localhost",
    user="root",
    port=3306,
    password="Sam512011",
    database="bw_mismatch_db"
)

cursor = connction.cursor()


sql = "SELECT id, min_bandwidth, tracking_err_j1, tracking_err_j2, tracking_err_j3, tracking_err_j4, tracking_err_j5, tracking_err_j6 FROM bw_mismatch_joints_data;"
cursor.execute(sql)
tracking_err_j2 = np.array([])
data = cursor.fetchall()

#%%
min_bandwidth = []
tracking_err_joints = [[] for _ in range(6)]

for _, row in enumerate(data):
    # print(row[1])
    # print(len(json.loads(row[0])))
    # tracking_err_j1 = np.append(tracking_err_j1, json.loads(row[0]))
    min_bandwidth.append(row[1])
    for i in range(6):
        tracking_err_joints[i].append(json.loads(row[i+2]))
# print(row)
# print(len(tracking_err_joints))
# tracking_err_j1 = tracking_err_j1.reshape((-1, len(json.loads(row[0]))))
# %%

tracking_err_joints = np.array(tracking_err_joints)
print(tracking_err_joints.shape)
tracking_err_joints = tracking_err_joints.transpose((1, 0, 2))
print(type(tracking_err_joints))
min_bandwidth = np.array(min_bandwidth)
# print((tracking_err_joints.shape))

# %%
print(type(tracking_err_joints))
output = []    
for i, bw in enumerate(min_bandwidth):
    arr = [0]*6
    arr[bw] = 1
    output.append(arr)
output = np.array(output)
print(tracking_err_joints.shape)
print(output.shape)
# %%

input_shape = (tracking_err_joints[0,:,:].shape) #length of row in array contouring_err
output_shape = (output.shape[1])

print(input_shape, output_shape)
# %%
