import os


# if folder does not exist, create it
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)