import os
import pickle


# Specify the directory containing the UCF50 dataset. 
DATASET_DIR = "dataset/UCF101"

#List all folders within the parent directory
folder_names = [folder for folder in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, folder))]

# # Print the folder names
for folder_name in folder_names:
    print(folder_name)

pickle.dump(folder_names,open('filenames.pkl','wb'))
