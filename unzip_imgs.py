import os 
import zipfile 
from PIL import Image 
import numpy as np 
import pandas as pd 
from tqdm import tqdm 

# X = [] 
# y_label = []

# zip_folders = ['part1.zip']
# path_to_folder = 'UCF_image_data'

labels = pd.read_csv("output_data/ucf_lat_lon_interpolated.csv")
labels = labels.set_index("merge_col")
# #labels['label_ordinal'] = pd.cut(labels.pred_post_pct_change, 15, labels=False)

# for zip_folder in zip_folders: 
#     file_name = os.path.join(path_to_folder, zip_folder)
#     with zipfile.ZipFile(file_name, "r") as zip_data:
#         content_list = zip_data.namelist()
#         unique_loc = 0
#         img_locs = set()
#         for i in tqdm(range(0, len(content_list), 1)):
#             name_file = content_list[i]
#             name_split = name_file.split('_')
#             loc = name_split[0]
#             view = name_split[1][0]
#             # skip if its a 4, as this is duplicate !
#             if int(view) == 4: continue 
#             if name_split[0] not in img_locs: 
#                 # open image 
#                 img_bytes = zip_data.open(name_file)
#                 img_data = Image.open(img_bytes)
#                 img_data.save("main_data/" + name_file)
#                 unique_loc += 1
#                 img_locs.add(name_split[0])

#         print("Unique Locations:", unique_loc, "in zip file", zip_folder)




files = os.listdir("main_data")
labels_idx = [int(f.split("_")[0]) for f in files]

y = labels.loc[labels_idx, "pred_post_pct_change"]
y = pd.cut(y, np.arange(y.min() - 0.01, y.max()+ 1, 1), labels=False)
y = y.astype("int")
y = np.array(y)

# create class folders 
for i in range(len(set(y))):
    os.mkdir("main_data/" + str(i))

# move imgs to where they belong 
import shutil 
for i, f in enumerate(files):
    shutil.move("main_data/" + f, "main_data/" + str(y[i]) + "/" + f)