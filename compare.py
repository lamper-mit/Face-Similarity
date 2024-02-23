#!/usr/bin/env python
import sys
import os
import shutil
import json
from deepface import DeepFace
from PIL import Image
import numpy as np
import base64
from io import BytesIO
from scipy.spatial.distance import cosine
from scipy.stats import zscore
# List of allowed image extensions
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.heic', '.tiff'}
scores = {}

if os.name == 'nt':
    windows = True

if not windows:
    import pyheif
    windows = False

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
def ensure_rgb_format(image_path):
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
            img.save(image_path)

def get_embedding(photo_path):
    if os.path.exists(photo_path):
        image = Image.open(photo_path)
    else:
        photo_base64 = photo_path
        image_bytes = base64.b64decode(photo_base64)
        image = Image.open(BytesIO(image_bytes))
    image_array = np.array(image)
    try:
        detected_faces = DeepFace.extract_faces(image_array, detector_backend = 'opencv')
    except ValueError:
        #print(f"No face detected in {photo_path}. Skipping...")
        print("No face detected")
        return None
    results = DeepFace.represent(img_path=image_array, model_name="VGG-Face", enforce_detection=True)
    embedding =  np.array(results[0]['embedding'])
    #norm = np.linalg.norm(embedding)
    if np.isnan(embedding).any():
        print(f"Warning: NaN values detected in embedding for {photo_path}")
    return embedding
def save_embeddings(embeddings, filepath):
    with open(filepath, 'w') as f:
        json.dump(embeddings, f)

# Function to load embeddings from a JSON file
def load_embeddings(filepath):
    with open(filepath, 'r') as f:
        embeddings = json.load(f)
    return embeddings
    
def calculate_similarity_scores(photo_path, source_embeddings):
    target_embedding = get_embedding(photo_path)
    if target_embedding is None:
        #print(f"Skipping {photo_path} due to no embedding.")
        print("no embedding")
        return None

    similarities = []
    for embedding in source_embeddings:
        similarity = 1 - cosine(np.array(embedding), np.array(target_embedding))
        similarities.append(similarity)
    #print(f"Similarities for {photo_path}: {similarities}")
    #print(f"Target embedding for {photo_path}: {target_embedding}")

    # Calculate the mean of the distances
    mean_similarity = np.mean(similarities)
    if np.isnan(mean_similarity):
        print(f"Warning: NaN similarity score for {photo_path}")
        return None
    return mean_similarity

def filter_embeddings_and_calculate_average_similarity(embeddings, outlier_threshold=0.45):
    """
    Remove outliers based on L2 norm and calculate the average similarity score among the remaining embeddings.

    Args:
    embeddings (list): A list of embeddings (numpy arrays).
    outlier_threshold (float): Threshold for determining outliers.

    Returns:
    tuple: A tuple containing the filtered embeddings and the average similarity score.
    """
    num_embeddings = len(embeddings)
    similarity_matrix = np.zeros((num_embeddings, num_embeddings))

    # Calculate pairwise L2 distances
    for i in range(num_embeddings):
        for j in range(i + 1, num_embeddings):
            similarity = 1 - cosine(embeddings[i],embeddings[j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    # Determine the mean distance for each embedding
    mean_similarities = np.mean(similarity_matrix, axis=1)
    # Identify embeddings with mean distance above the threshold
    non_outliers = mean_similarities > outlier_threshold
    # Filter out outliers
    filtered_embeddings = [emb for idx, emb in enumerate(embeddings) if non_outliers[idx]]

    # Calculate average similarity (distance) among the remaining embeddings
    average_similarity = np.mean([similarity_matrix[i, j] for i in range(num_embeddings) 
                                  for j in range(i+1, num_embeddings) if non_outliers[i] and non_outliers[j]])

    return filtered_embeddings, average_similarity
def verify_and_copy(source_directory, reference_directory, target_directory=None, cutoff=0.45):
    """
    Verify images in the source directory against the reference directory,
    and copy images that meet the similarity cutoff to the target directory.

    Args:
    source_directory (str): Path to the source directory containing images to be verified.
    reference_directory (str): Path to the reference directory containing reference images.
    target_directory (str): Path to the target directory where matching images will be copied.
    cutoff (float, optional): The similarity score cutoff. If None, calculated dynamically.
    """
    # making sure pipeline continues to run
    if not os.path.exists(reference_directory):
        return []
    embeddings_file = os.path.join(reference_directory, 'reference_embeddings.json')
    reference_embeddings = []
    # Check if the embeddings file exists and load it
    if os.path.exists(embeddings_file):
        print("Loading existing embeddings...")
        reference_embeddings = load_embeddings(embeddings_file)
        reference_embeddings = [np.array(emb) for emb in reference_embeddings]  # Convert lists back to numpy arrays
    else:
        # Calculate embeddings for reference images
        print("Calculating new embeddings...")
        for file in os.listdir(reference_directory):
            if is_image_file(file):
                file_path = os.path.join(reference_directory, file)
                # Check and convert HEIC files if necessary
                if file.lower().endswith('.heic') and not windows:
                    # Uncomment the next line if you've implemented the convert_heic_to_jpg function
                    file_path = convert_heic_to_jpg(file_path)
                    
                ensure_rgb_format(file_path)
                
                embedding = get_embedding(file_path)
                if embedding is not None:
                    reference_embeddings.append(embedding.tolist())  # Convert numpy array to list for JSON serialization

        # Save the newly calculated embeddings
        save_embeddings(reference_embeddings, embeddings_file)
        reference_embeddings = [np.array(emb) for emb in reference_embeddings]  # Convert lists back to numpy arrays for further processing
    filtered_reference_embeddings, average_similarity = filter_embeddings_and_calculate_average_similarity(reference_embeddings)

    #cutoff = (1 + 0.20) * average_similarity if cutoff is None else cutoff
    
    scores = {}
    #check to see whether the directory is a path or a list
    if isinstance(source_directory, (str, bytes, os.PathLike)):
        for file in os.listdir(source_directory):
            if is_image_file(file):
                file_path = os.path.join(source_directory, file)
                ensure_rgb_format(file_path)
                
                similarity_score = calculate_similarity_scores(file_path, filtered_reference_embeddings)
                if similarity_score is None:
                    continue
                scores[file_path] = similarity_score
    else:
        for index,image in enumerate(source_directory):
            similarity_score = calculate_similarity_scores(image, filtered_reference_embeddings)
            if similarity_score is not None:
                # Assuming you want to use the index in the key or for some other purpose
                scores[f"Image {index}"] = similarity_score
    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    if cutoff is not None:
        filtered_scores = {k: v for k, v in sorted_scores.items() if v > cutoff}
        if isinstance(source_directory, list):
        # Initialize an empty list to hold base64 strings that meet the cutoff
            filtered_base64_strings = []
        
            # Iterate through the filtered scores
            for key in filtered_scores.keys():
                # Extract the index from the key (assuming keys are in the format "Image {index}")
                index = int(key.split()[1])
                # Append the corresponding base64 string from the original list
                filtered_base64_strings.append(source_directory[index])
        
            # Return the list of filtered base64 strings
            return filtered_base64_strings
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

    return output_data
if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: ./compare.py <source_directory> <target_directory> <reference_directory> [distance_cutoff],/n Check your parameters and try again.")
        sys.exit(1)

    source_directory = sys.argv[1]
    reference_directory = sys.argv[2]
    target_directory = sys.argv[3]
    distance_cutoff = float(sys.argv[4]) if len(sys.argv) == 5 else None

    verify_and_copy(source_directory, reference_directory, target_directory, distance_cutoff)
