#!/usr/bin/env python
import sys
import os
import shutil
import json
from deepface import DeepFace
from PIL import Image
import pyheif
import numpy as np
from deepface.commons import functions

# List of allowed image extensions
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif','.heic', '.tiff'}
scores = {}
composite_norm = 0
photo_norm = 0
def convert_heic_to_jpg(heic_path):
    heif_file = pyheif.read(heic_path)
    image = Image.frombytes(
        heif_file.mode, 
        heif_file.size, 
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    jpg_path = heic_path.replace('.heic', '.jpg')
    image.save(jpg_path, "JPEG")
    return jpg_path
def is_image_file(filename):
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
    composite_mean = np.mean(embeddings, axis=0)
    composite_norm = np.linalg.norm(composite_mean)
    return composite_mean/composite_norm

def verify_with_composite(photo_path, composite_embedding, cutoff):
    photo_embedding = get_embedding(photo_path)
    if photo_embedding is None:
        return False
    photo_norm = np.linalg.norm(photo_embedding)
    photo_embedding = photo_embedding/photo_norm
    distance = np.linalg.norm(photo_embedding - composite_embedding)
    scores[photo_path] = distance
    return distance < cutoff

def verify_and_copy(source_directory, target_directory, reference_directory, cutoff=0.4):
    composite_embedding = calculate_composite_from_directory(reference_directory)
    
    for file in os.listdir(source_directory):
        if is_image_file(file):
            file_path = os.path.join(source_directory, file)
            
            # Check if the file is a .heic and convert it to .jpg if it is
            if file_path.endswith('.heic'):
                file_path = convert_heic_to_jpg(file_path)
                os.remove(file_path.replace('.jpg', '.heic'))
            
            verify_with_composite(file_path, composite_embedding, cutoff)
    
    # Sort the scores from most to least similar and filter by cutoff
    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1]))
    filtered_scores = {k: v for k, v in sorted_scores.items() if v < cutoff}
    
    # Copy the images in order of their similarity scores
    for photo_path in filtered_scores.keys():
        if os.path.exists(photo_path):
            file_name = os.path.basename(photo_path)
            target_path = os.path.join(target_directory, file_name)
            shutil.copy(photo_path, target_path)
    
    # Create the output dictionary with cutoff and scores
    output_data = {
        "cutoff": cutoff,
        "scores": sorted_scores
    }
    
    with open(os.path.join(target_directory, 'similarity_scores.json'), 'w') as json_file:
        json.dump(output_data, json_file, indent=4)
if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: ./verify_photos.py <source_directory> <target_directory> <reference_directory> [distance_cutoff],/n Check your parameters and try again.")
        sys.exit(1)

    source_directory = sys.argv[1]
    target_directory = sys.argv[2]
    reference_directory = sys.argv[3]
    distance_cutoff = float(sys.argv[4]) if len(sys.argv) == 5 else 0.60

    verify_and_copy(source_directory, target_directory, reference_directory, distance_cutoff)
