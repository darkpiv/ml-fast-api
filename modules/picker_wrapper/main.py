import pickle


def load_model(path: str):
    return pickle.load(open(path, 'rb'))
