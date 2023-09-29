#!/usr/bin/env python
import sys
import os
import shutil
import json
from deepface import DeepFace
import numpy as np
from deepface.commons import functions

# List of allowed image extensions
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
scores = {}
def is_image_file(filename):
    """Check if the file is an image based on its extension."""
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

def get_embedding(photo_path):

    try:
        detected_faces = functions.extract_faces(photo_path, detector_backend = 'opencv')
    except ValueError:
        print(f"No face detected in {photo_path}. Skipping...")
        return None
    results = DeepFace.represent(img_path=photo_path, model_name="VGG-Face", enforce_detection=True)
    return results[0]['embedding']

def calculate_composite_from_directory(directory):
    embeddings = []
    for file in os.listdir(directory):
        if is_image_file(file):
            file_path = os.path.join(directory, file)
            embedding = get_embedding(file_path)
            if embedding is not None: 
                embeddings.append(embedding)
        else:
            print(f"Ignoring non-image file: {file}")
    return np.mean(embeddings, axis=0)

def verify_with_composite(photo_path, composite_embedding):
    photo_embedding = get_embedding(photo_path)
    if photo_embedding is None:
        return False
    distance = np.linalg.norm(photo_embedding - composite_embedding)
    scores[photo_path] = distance
    return distance < 0.4

def verify_and_copy(source_directory, target_directory, reference_directory):
    composite_embedding = calculate_composite_from_directory(reference_directory)
    for file in os.listdir(source_directory):
        if is_image_file(file):
            source_path = os.path.join(source_directory, file)
            if verify_with_composite(source_path, composite_embedding):
                target_path = os.path.join(target_directory, file)
                shutil.copy(source_path, target_path)
    with open(os.path.join(target_directory, 'similarity_scores.json'), 'w') as json_file:
        json.dump(scores, json_file, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: ./verify_photos.py <source_directory> <target_directory> <reference_directory>")
        sys.exit(1)

    source_directory = sys.argv[1]
    target_directory = sys.argv[2]
    reference_directory = sys.argv[3]

    verify_and_copy(source_directory, target_directory, reference_directory)
