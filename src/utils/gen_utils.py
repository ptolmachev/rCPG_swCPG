import pandas as pd
import os
import re
from pathlib import Path

def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent.parent

def create_dir_if_not_exist(path):
    try:
        os.makedirs(path, exist_ok=False)
    except:
        pass
    return None

def get_folders(root_folder, pattern):
    folders_all = os.listdir(root_folder + '/')
    folders = []
    for i, folder in enumerate(folders_all):
        m = re.search(pattern, str(folder))
        if m is not None:
            folders.append(folder)
    return folders

def get_files(root_folder, pattern):
    folders_and_files = os.listdir(root_folder + '/')
    files = []
    for i, el in enumerate(folders_and_files):
        if os.path.isfile(root_folder  + '/' + el):
            m = re.search(pattern, str(el))
            if m is not None:
                files.append(el)
    return files

def get_postfix(inh_NTS, inh_KF):
    if inh_NTS == 1 and inh_KF == 1:
        postfix = 'intact'
    elif inh_NTS == 2 and inh_KF == 1:
        postfix = 'inh_NTS'
    elif inh_NTS == 1 and inh_KF == 2:
        postfix = 'inh_KF'
    elif inh_NTS == 2 and inh_KF == 2:
        postfix = 'inh_NTS_inh_KF'
    elif inh_NTS == 0 and inh_KF == 1:
        postfix = 'disinh_NTS'
    elif inh_NTS == 1 and inh_KF == 0:
        postfix = 'disinh_KF'
    elif inh_NTS == 0 and inh_KF == 0:
        postfix = 'disinh_NTS_disinh_KF'
    elif inh_NTS == 0 and inh_KF == 2:
        postfix = 'disinh_NTS_inh_KF'
    elif inh_NTS == 2 and inh_KF == 0:
        postfix = 'inh_NTS_disinh_KF'
    return postfix

def fill_nans(data):
    df = pd.DataFrame(data)
    data_new = df.fillna(df.mean()).values
    return data_new

# if __name__ == '__main__':
#     root_path = get_project_root()
#     x = 1


