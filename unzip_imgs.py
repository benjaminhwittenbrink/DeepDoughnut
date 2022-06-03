import argparse
import os 
import pickle 
import zipfile 
from PIL import Image 
import numpy as np 
import pandas as pd 
from tqdm import tqdm 


parser = argparse.ArgumentParser()

parser.add_argument('--folder', default='sample', type=str, help='Image data location.')
args = parser.parse_args()

X = [] 
y_label = []

zip_folders = os.listdir("UCF_image_data")
zip_folders = sorted([f for f in zip_folders if ".zip" in f])
#zip_folders = ['part1.zip', 'part2.zip', 'part3.zip']
#zip_folders = ['part9.zip']
#zip_folders = zip_folders[:-1]
print(zip_folders)
# zip_folders = ['part1.zip']
path_to_folder = 'UCF_image_data'

labels = pd.read_csv("output_data/ucf_lat_lon_interpolated.csv")
labels = labels.set_index("merge_col")
#labels['label_ordinal'] = pd.cut(labels.pred_post_pct_change, 15, labels=False)

to_folder = args.folder

for zip_folder in zip_folders: 
    file_name = os.path.join(path_to_folder, zip_folder)
    with zipfile.ZipFile(file_name, "r") as zip_data:
        content_list = zip_data.namelist()
        unique_loc = 0
        img_locs = set()
        for i in tqdm(range(0, len(content_list), 1)):
            name_file = content_list[i]
            name_split = name_file.split('_')
            loc = name_split[0]
            view = name_split[1][0]
            # skip if its a 4, as this is duplicate !
            if int(view) in [0, 4] : continue 
            if name_split[0] not in img_locs: 
                # open image 
                img_bytes = zip_data.open(name_file)
                img_data = Image.open(img_bytes)
                img_data.save(to_folder  + "/" + name_file)
                unique_loc += 1
                img_locs.add(name_split[0])

        print("Unique Locations:", unique_loc, "in zip file", zip_folder)


files = os.listdir(to_folder)
labels_idx = [int(f.split("_")[0]) for f in files]

y_cont = labels.loc[labels_idx, "pred_post_pct_change"]
y_ord = pd.cut(y_cont, np.arange(y_cont.min() - 0.01, y_cont.max()+ 1, 1), labels=False)
y_ord = y_ord.astype("category").cat.rename_categories([i for i in range(len(set(y_ord)))])
y_ord = y_ord.astype("int")
y_ord = np.array(y_ord)
print(y_ord)
print(pd.Series(y_ord).value_counts())
 
# y_cont_dict = dict(zip(files, y_cont))
# with open("part1" + "_scores_dict.pickle", "wb") as f: 
#     pickle.dump(y_cont_dict, f)

# # create class folders 
# for i in range(len(set(y_ord))):
#     os.mkdir(to_folder + "/" + str(i))

# # move imgs to where they belong 
# import shutil 
# for i, f in enumerate(files):
#     shutil.move(to_folder + "/" + f, to_folder + "/" + str(y_ord[i]) + "/" + f)
 

# folder = "main_data2_east"
# for c in os.listdir(folder):
#     cfiles = os.listdir(folder + "/" + str(c))
#     random_files = np.random.choice(cfiles, int(len(cfiles) * 0.15), replace = False)
#     val_folder = folder + "_val"
#     os.mkdir(val_folder + "/" + str(c))
#     for f in random_files:
#         shutil.move(folder + "/" + str(c) + "/" + f, val_folder + "/" + str(c) + "/" + f) 
