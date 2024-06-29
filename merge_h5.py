import os
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
import h5py

class Hdf5:

    def __init__(self, fname, lib='h5py', overwrite=False):
        self.fname = fname
        self.file = None
        if overwrite and os.path.exists(fname):
            os.remove(fname)

    def add(self, key, value):
        with h5py.File(self.fname, 'a', libver='latest') as f:
            if key in f.keys():
                print(f"{key} already existed in {self.fname}, skipping...")
            else:
                f.create_dataset(
                    key,
                    data=value,
                    maxshape=value.shape,
                    compression='lzf',
                    shuffle=True,
                    track_times=False,
                    # track_order=False,
                )

    def add_subset(self, key, link):
        with h5py.File(self.fname, 'a', libver='latest') as f:
            f[key] = h5py.ExternalLink(link, '/')

    def get(self, key):
        if self.file is None:
            self.file = h5py.File(self.fname, 'r', libver='latest')
        if '/' in key:
            value = self.file
            for k in key.split('/'):
                value = value[k]
        else:
            value = self.file[key]
        return value

    def load(self):
        if self.file is None:
            self.file = h5py.File(self.fname, 'r', libver='latest')
        return self.file

    @property
    def keys(self):
        if self.file is None:
            self.file = h5py.File(self.fname, 'r', libver='latest')
        return sorted(list(self.file.keys()))

    def iter_keys(self, key):
        if self.file is None:
            self.file = h5py.File(self.fname, 'r', libver='latest')

        value = self.file
        if '/' in key:
            for k in key.split('/'):
                value = value[k]

        return sorted(list(value.keys()))


class Merge:

    def __init__(self, ROOT_PATH, output_dir):
        folder_tree = dict()
        self.root_path = ROOT_PATH
        files = glob(f"{ROOT_PATH}/**/*/cache.h5", recursive=True)
        self.cache = Hdf5(output_dir, overwrite=True)

        for file in files:
            folder_tree[file] = Hdf5(file)
                
        self.tree = folder_tree


    def runner(self):
        # for folder, cache in tqdm(self.tree.items(), total=len(self.tree), position=1):
        #     # for key in tqdm(cache.keys, position=2, leave=False):
        #     for key in cache.keys:
        #         self.cache.add(folder.replace('cache.h5', key), cache.get(key))

        #     cache.file.close()
        for path, cache in tqdm(self.tree.items(), total=len(self.tree), position=1):
            
            self.cache.add_subset(path.replace('cache.h5', ''), path)
            del cache


Merge(ROOT_PATH='data/HDTF/', output_dir='data/HDTF/data.h5').runner()
Merge(ROOT_PATH='data/LRS2/', output_dir='data/LRS2/data.h5').runner()

# Merge(ROOT_PATH='data/finetune/', output_dir='data/finetune/data.h5').runner()
