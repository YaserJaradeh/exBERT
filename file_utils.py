from typing import Any
import h5py
import pickle


def serialize_data(data: Any, path: str):
    pickle.dump(data, open(path, "wb"))


def deserialize_data(path: str):
    return pickle.load(open(path, "rb"))


def serialize_data_hdf5(data: Any, path: str):
    with h5py.File(f"{path.replace('.pkl', '.hdf5')}", "w") as f:
        f.create_dataset('mydataset', (len(data),))


def deserialize_data_hdf5(path: str):
    with h5py.File(f"{path.replace('.pkl', '.hdf5')}", "r") as f:
        return f['mydataset']
