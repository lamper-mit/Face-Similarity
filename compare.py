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
from scipy.spatial import distance
from scipy.stats import zscore
# List of allowed image extensions
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.heic', '.tiff'}
scores = {}

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
def identify_outliers(embeddings):
    """
    Identify outliers in a list of embeddings using Z-score.
    Returns a list of boolean values, where True indicates that the corresponding embedding is an outlier.
    """
    means = np.mean(embeddings, axis=0)
    std_devs = np.std(embeddings, axis=0)
    if np.any(std_devs == 0):
        print("Warning: Zero standard deviation detected.")
    
    z_scores = np.abs((embeddings - means) / std_devs)
    max_z_scores = np.max(z_scores, axis=1)
    
    # Consider an embedding as an outlier if any of its dimensions has a Z-score greater than 2.
    # This threshold can be adjusted as needed.
    outliers = max_z_scores > 2
    
    return outliers

def ensure_rgb_format(image_path):
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
            img.save(image_path)

def get_embedding(photo_path):
    try:
        detected_faces = functions.extract_faces(photo_path, detector_backend = 'opencv')
    except ValueError:
        print(f"No face detected in {photo_path}. Skipping...")
        return None
    results = DeepFace.represent(img_path=photo_path, model_name="VGG-Face", enforce_detection=True)
    embedding =  np.array(results[0]['embedding'])
    if np.isnan(embedding).any():
        print(f"Warning: NaN values detected in embedding for {photo_path}")
    return embedding 

def calculate_similarity_scores(photo_path, source_embeddings):
    target_embedding = get_embedding(photo_path)
    if target_embedding is None:
        print(f"Skipping {photo_path} due to no embedding.")
        return None

    distances = []
    for embedding in source_embeddings:
        distance = np.linalg.norm(np.array(embedding) - np.array(target_embedding))
        distances.append(distance)

    # Calculate the mean of the distances
    mean_distance = np.mean(distances)
    if np.isnan(mean_distance):
        print(f"Warning: NaN similarity score for {photo_path}")
        return None
    return mean_distance


def verify_and_copy(source_directory, target_directory, reference_directory, cutoff=0.4):
    source_embeddings = []
    for file in os.listdir(source_directory):
        if is_image_file(file):
            file_path = os.path.join(source_directory, file)
            if file_path.endswith('.heic'):
                file_path = convert_heic_to_jpg(file_path)
                os.remove(file_path.replace('.jpg', '.heic'))
            ensure_rgb_format(file_path)
            embedding = get_embedding(file_path)
            if embedding is not None:
                source_embeddings.append(np.array(embedding))
    outliers = identify_outliers(source_embeddings)
    source_embeddings = [embedding for i, embedding in enumerate(source_embeddings) if not outliers[i]]

    for file in os.listdir(reference_directory):
        if is_image_file(file):
            file_path = os.path.join(reference_directory, file)
            similarity_score = calculate_similarity_scores(file_path, source_embeddings)
            if similarity_score is None:
                continue
            scores[file_path] = similarity_score
    
    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1]))
    filtered_scores = {k: v for k, v in sorted_scores.items() if v < cutoff}
    
    for photo_path in filtered_scores.keys():
        if os.path.exists(photo_path):
            file_name = os.path.basename(photo_path)
            target_path = os.path.join(target_directory, file_name)
            shutil.copy(photo_path, target_path)
    
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
