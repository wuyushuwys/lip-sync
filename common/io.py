"""File IO helper."""
import os
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
