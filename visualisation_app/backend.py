import numpy as np
from itertools import combinations

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, ".."))
if root_path not in sys.path:
    sys.path.append(root_path)
sys.path.append(os.path.join(root_path, "ImageBind"))

import imagebind
import torch
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


def init_imagebind_model():

    # device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # # Instantiate model
    # model = imagebind_model.imagebind_huge(pretrained=True)
    # model.eval()
    # model.to(device)

    model = None
    device = None

    return model, device


def get_embeddings(model, device, selected_objects):

    # TODO dynamically get the right ones
    # text_list=["A dog.", "A car", "A bird", "A piano"]
    # image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg", ".assets/piano_image.jpg"]
    # audio_paths=[".assets/dog_audio.wav", ".assets/car_audio.wav", ".assets/bird_audio.wav", ".assets/piano_audio.wav"]

    # # Load data
    # inputs = {
    #     ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    #     ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    #     ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
    # }

    # with torch.no_grad():
    #     embeddings = model(inputs)

    # print(embeddings)
    

    text_embeddings = {
        'bird': [1, 5],
        'car': [8, 2],
        'dog': [2, 4],
        'piano': [5, 8]
    } 


    audio_embeddings = {
        'bird': [1, 7],
        'car': [8, 2],
        'dog': [2, 3],
        'piano': [5, 9]
    } 

    vision_embeddings = {
        'bird': [1, 7],
        'car': [8, 2],
        'dog': [2, 3],
        'piano': [5, 9]
    }

    embeddings = {
        "Text": text_embeddings,
        "Audio" : audio_embeddings,
        "Vision" : vision_embeddings
    }

    return embeddings


def compute_pairwise_dot_products(embeddings, selected_objects):
    """
    Logic for cross-modal alignment calculation.
    """
    if not selected_objects:
        return {}

    mod_names = list(embeddings.keys())
    mod_pairs = list(combinations(mod_names, 2))
    results = {}

    for obj in selected_objects:
        obj_results = {}
        for mod1, mod2 in mod_pairs:
            # Safely check if object exists in both modalities
            if obj in embeddings.get(mod1, {}) and obj in embeddings.get(mod2, {}):
                vec1 = np.array(embeddings[mod1][obj])
                vec2 = np.array(embeddings[mod2][obj])
                
                dot_prod = np.dot(vec1, vec2)
                pair_key = f"{mod1} * {mod2}"
                obj_results[pair_key] = round(float(dot_prod), 4)
        
        if obj_results:
            results[obj] = obj_results
            
    return results