import numpy as np
from itertools import combinations

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

assets_path = os.path.join(root_path, ".assets")

import imagebind
import torch
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

import json

MODALITIES = ['audio', 'text', 'vision']

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

    # text_list=[f"A {obj}" for obj in selected_objects]
    # image_paths=[f"my_assets/{obj}_image.jpg" for obj in selected_objects]
    # audio_paths=[f"my_assets/{obj}_audio.wav" for obj in selected_objects]

    # # Load data
    # inputs = {
    #     ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    #     ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    #     ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
    # }

    # with torch.no_grad():
    #     embeddings = model(inputs)

    # device = "cuda:0" if torch.cuda.is_available() else "cpu"

    embeddings = {}

    for mod in MODALITIES:

        with open(f'{assets_path}/{mod}_embeddings.json', 'rb') as f:
            # embeddings[mod] = torch.load(f, map_location=torch.device(device))
            embeddings[mod] = json.load(f)

    return embeddings


def compute_dot_products(embeddings, selected_objects):
    """
    Logic for cross-modal alignment calculation.
    """
    if not selected_objects:
        return {}

    products = {}

    audio_tensor = torch.tensor([v for k,v in embeddings['audio'].items() if k in selected_objects])
    text_tensor = torch.tensor([v for k,v in embeddings['text'].items() if k in selected_objects])
    vision_tensor = torch.tensor([v for k,v in embeddings['vision'].items() if k in selected_objects])
    

    products['VT'] = torch.softmax(vision_tensor @ text_tensor.T, dim=1)
    products['AT'] = torch.softmax(vision_tensor @ audio_tensor.T, dim=1)
    products['VA'] = torch.softmax(vision_tensor @ audio_tensor.T, dim=1)
    
    return products