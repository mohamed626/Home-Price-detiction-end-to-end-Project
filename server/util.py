import json
import pickle
import  numpy as np

__locations= None
__data_columns= None
__model= None


def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    nc = np.zeros(len(__data_columns))
    nc[0] = sqft
    nc[1] = bath
    nc[2] = bhk
    if loc_index >= 0:
        nc[loc_index] = 1

    return round(__model.predict([nc])[0], 2)


def get_location_names():
    return __locations


def load_saved_artifacts ():
    print("load start")
    global __locations
    global __data_columns

    with open("./artifacts/columns.json", 'r') as f:
        __data_columns=json.load(f)['data_columns']
        __locations = __data_columns[3:]

    global __model
    with open("./artifacts/linear_regrision_model_to_prdict_price.pickle", 'rb') as f:
        __model = pickle.load(f)
    print("sucsses load")


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st phase jp nagar', 1000, 3, 3))
    print(get_estimated_price('1st phase jp nagar', 1000, 2, 2))
    print(get_estimated_price('kalhala', 1000, 2, 2)) #other locaton