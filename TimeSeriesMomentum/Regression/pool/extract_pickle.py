#%%
import os
import shutil
import pandas as pd
import numpy as np

current_dir = os.getcwd()
all_file_list = os.listdir(current_dir)
ind_folder = ['fr_ol', 'fr_nonol', 'hb_ol', 'hb_nonol','hb_hprxnonol','fr_hprxnonol']
new_dir = "./result/"

if not os.path.exists(new_dir):
    os.makedirs(new_dir)

for folder in ind_folder:
    print("running "+folder)
    for year_folder in range(2001, 2016):
        print(year_folder)
        old_dir = os.path.join(folder, str(year_folder))
        for file in os.listdir(old_dir):
            if '.pickle' in file:
                shutil.copyfile(os.path.join(old_dir, file), os.path.join(new_dir, file))
