import numpy as np
import pandas as pd
import time

def bytesToString(bytesObject):
    if hasattr(bytesObject, 'decode'):
        return bytesObject.decode()
    return bytesObject

def convertToNumpy(data):
    """
    Converts data to numpy if the data is an object of pd.DataFrame or a list
    of lists. Otherwise, raise ValueError.
    """
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, pd.DataFrame):
        return data.to_numpy()
    elif isinstance(data, list):
        data = np.array(data)
        if len(data) == 2:
            return data
        else:
            raise ValueError(f"Expected a 2D list as input")
    raise ValueError(f"type {type(data)} not supported")
