#!/usr/bin/env python

# Import necessary libraries
import sys
import os
import shutil
import json
from deepface import DeepFace
import numpy as np
from deepface.commons import functions

# Define a set of allowed image extensions
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

# Dictionary to store similarity scores for each image
scores = {}

def is_image_file(filename):
    """
    Check if the given filename is an image based on its extension.
    
    Args:
    - filename (str): Name of the file to check.

    Returns:
    - bool: True if the file is an image, False otherwise.
    """
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

def get_embedding(photo_path):
    """
    Extract the face embedding for a given photo.
    
    Args:
    - photo_path (str): Path to the photo.

    Returns:
    - array: Embedding of the detected face or None if no face is detected.
    """
    try:
        # Extract faces from the photo
        detected_faces = functions.extract_faces(photo_path, detector_backend = 'opencv')
    except ValueError:
        print(f"No face detected in {photo_path}. Skipping...")
        return None
    # Get the embedding for the detected face
    results = DeepFace.represent(img_path=photo_path, model_name="VGG-Face", enforce_detection=True)
    return results[0]['embedding']

def calculate_composite_from_directory(directory):
    """
    Calculate the average face embedding for all images in a directory.
    
    Args:
    - directory (str): Path to the directory containing images.

    Returns:
    - array: Average face embedding for the directory.
    """
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
    return composite_mean/np.linalg.norm(composite_mean)

def verify_with_composite(photo_path, composite_embedding, cutoff):
    """
    Verify if a photo is similar to a composite embedding based on a cutoff value.
    
    Args:
    - photo_path (str): Path to the photo to verify.
    - composite_embedding (array): The composite (average) embedding to compare against.
    - cutoff (float): The similarity threshold.

    Returns:
    - bool: True if the photo is similar to the composite, False otherwise.
    """
    photo_embedding = get_embedding(photo_path)
    if photo_embedding is None:
        return False
    photo_embedding = photo_embedding/np.linalg.norm(photo_embedding)
    distance = np.linalg.norm(photo_embedding - composite_embedding)
    scores[photo_path] = distance
    return distance < cutoff

def verify_and_copy(source_directory, target_directory, reference_directory, cutoff=0.4):
    """
    Verify photos in the source directory against a reference directory and copy similar photos to the target directory.
    
    Args:
    - source_directory (str): Directory containing photos to verify.
    - target_directory (str): Directory to copy similar photos to.
    - reference_directory (str): Directory containing reference photos to generate the composite embedding.
    - cutoff (float, optional): Similarity threshold. Defaults to 0.4.
    """
    # Calculate the composite embedding for the reference directory
    composite_embedding = calculate_composite_from_directory(reference_directory)
    
    # Verify each photo in the source directory
    for file in os.listdir(source_directory):
        if is_image_file(file):
            source_path = os.path.join(source_directory, file)
            verify_with_composite(source_path, composite_embedding, cutoff)
    
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
    
    # Save the scores to a JSON file
    with open(os.path.join(target_directory, 'similarity_scores.json'), 'w') as json_file:
        json.dump(output_data, json_file, indent=4)

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: ./verify_photos.py <source_directory> <target_directory> <reference_directory> [distance_cutoff]\nCheck your parameters and try again.")
        sys.exit(1)

    # Parse command line arguments
    source_directory = sys.argv[1]
    target_directory = sys.argv
