import numpy as np
from itertools import combinations

def get_embeddings():

    # TODO dynamically get the right ones

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