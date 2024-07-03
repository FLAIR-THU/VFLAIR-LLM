import os

def get_model_folder():
    model_folder = os.getenv('MODEL_FOLDER')
    if model_folder is None:
        raise ValueError('MODEL_FOLDER env must not be empty and should contain /')
    return model_folder