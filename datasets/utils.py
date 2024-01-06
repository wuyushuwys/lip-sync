import os
from common import Hdf5


def load_from_folder(folder_tree, folder, mode, ext):
    if mode == 'image':
        frame_list = filter(lambda x: x.endswith(ext), os.listdir(folder))
        if len(frame_list) > 0:  # only load if frame file exist
            folder_tree[folder] = sorted(frame_list)
    elif mode == 'h5':
        cache_path = os.path.join(folder, 'cache.h5')
        if os.path.exists(cache_path):  # only load if cache file exist
            folder_tree[folder] = Hdf5(cache_path)
    else:
        raise NotImplementedError(f"{mode} not supported")
    return folder_tree
