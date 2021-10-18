import pickle


# Load pickle file
def load_pkl(target_path):
    with open(target_path, 'rb') as handle:
        return pickle.load(handle)


# Dump data to pickle file
def save_pkl(target_path, data):
    with open(target_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
