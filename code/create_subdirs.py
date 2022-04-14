import os

def create_subdirs(path, sub_dirs):

    for sub_dir in sub_dirs:
        path_to_create = os.path.join(path, sub_dir)
        if not os.path.exists(path_to_create):
            os.makedirs(path_to_create)
            print(f"created path {path_to_create}")