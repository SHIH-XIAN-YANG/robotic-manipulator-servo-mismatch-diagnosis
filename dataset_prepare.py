import numpy as np


file = 'tracking_err_Z.txt'


file_paths = ['C:/Users/Samuel/Desktop/mismatch_diagnal_NN/NN_data/data/dataset/',
              'C:/Users/Samuel/Desktop/mismatch_diagnal_NN/NN_data/data/dataset_5000/',
              'C:/Users/Samuel/Desktop/mismatch_diagnal_NN/NN_data/data/dataset_10000/',
              'C:/Users/Samuel/Desktop/mismatch_diagnal_NN/NN_data/data/dataset2/']

save_path = 'C:/Users/Samuel/Desktop/mismatch_diagnal_NN/NN_data/train_data/'


# stack_data = np.stack([np.loadtxt(file_path+file, delimiter=',') for file_path in file_paths], axis=0)

data_list = []
for file_path in file_paths:
    data = np.loadtxt(file_path+file, delimiter=',')
    data_list.append(data)
# data_array = np.array(data_list)
# print(data_array.shape)

# Stack the data together
stacked_data = np.vstack(data_list)



output_file = save_path+file
np.savetxt(output_file, stacked_data)