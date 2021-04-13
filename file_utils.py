from typing import Any
import pickle


def serialize_data(data: Any, path: str):
    pickle.dump(data, open(path, "wb"))


def deserialize_data(path: str):
    return pickle.load(open(path, "rb"))
