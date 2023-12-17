"""File IO helper."""
import os
import h5py
import numpy as np


class Hdf5:

    def __init__(self, fname, lib='h5py', overwrite=False):
        self.fname = fname
        self.lib = lib
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

    def get(self, key):
        if not self.file:
            self.file = h5py.File(self.fname, 'r', libver='latest')
        return self.file[key]

    @property
    def keys(self):
        with h5py.File(self.fname, mode='r', libver='latest') as f:
            return list(f.keys())
