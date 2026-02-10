import logging
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from utils.loader import load_image, downsample, flatten_data, stack_data
from PIL import Image



def athlete_3m(args, seed=42):
    """
    Loads and preprocesses Variant-MNIST image data for training.
    
    This function:
    1. Reads CSV files for image paths.
    2. Loads images and constructs feature tensors.
    3. Splits the dataset into training and validation sets.
    4. Downsamples images from 28x28 to 14x14.
    5. Flattens images and tabular data.
    6. Returns processed train and test datasets.
    
    Args:
        seed (int): Random seed for train/test split.
        data_source (str): Path to the dataset directory.

    Returns:
        dict: A dictionary containing train and test data.
    """
    
    X1, X21, X22, X23 = [], [], [], []
    Z21, Z22, Z23, Z24, Z25 = [], [], [], [], []
    X1 = np.load("./dataset/athlete_3m/X1_imagebind_embeddings.npy")
    X21 = np.load("./dataset/athlete_3m/X21.npy")
    X22 = np.load("./dataset/athlete_3m/X22.npy")
    X23 = np.load("./dataset/athlete_3m/X23.npy")
    Z21 = np.load("./dataset/athlete_3m/Z21.npy")
    Z22 = np.load("./dataset/athlete_3m/Z22.npy")
    Z23 = np.load("./dataset/athlete_3m/Z23.npy")
    Z24 = np.load("./dataset/athlete_3m/Z24.npy")
    Z25 = np.load("./dataset/athlete_3m/Z25.npy")
    Z_all = np.load("./dataset/athlete_3m/Z_all_5_8.npy")

    
    # Dictionary to store original variables
    variables = {"X1": X1/255, "X21": X21, "X22": X22, "X23": X23, "Z21": Z21, "Z22": Z22, "Z23": Z23, "Z24": Z24, "Z25": Z25, "Z_all": Z_all}
    train_val_split = {}
    for key, value in variables.items():
        train_data, val_data = train_test_split(value, test_size=0.1, random_state=seed)
        train_val_split[f"{key}_train"] = train_data
        train_val_split[f"{key}_val"] = val_data

    # Log train/validation set shapes
    logging.info("Training set shapes:")
    for key in ["X1_train", "X21_train", "X22_train", "X23_train", "Z21_train", "Z22_train", "Z23_train", "Z24_train", "Z25_train", "Z_all_train"]:
        logging.info(f"  {key}: {train_val_split[key].shape}")

    logging.info("Validation set shapes:")
    for key in ["X1_val", "X21_val", "X22_val", "X23_val", "Z21_val", "Z22_val", "Z23_val", "Z24_val", "Z25_val", "Z_all_val"]:
        logging.info(f"  {key}: {train_val_split[key].shape}")
        
    train_data = stack_data(train_val_split, "train", args.case_id)
    val_data = stack_data(train_val_split, "val", args.case_id)
    all_data = np.concatenate([train_data, val_data], axis=0)
    input_dims = [train_val_split["X1_train"].shape[1], train_val_split["X21_train"].shape[1], train_val_split["X22_train"].shape[1], train_val_split["X23_train"].shape[1]]
    
    logging.info(f"Final Train Data Shape: {train_data.shape}")
    logging.info(f"Final Test Data Shape: {val_data.shape}")
    logging.info(f"Final All Data Shape: {all_data.shape}")
    logging.info(f"Input dimensions are: {input_dims}")
    
    # Define the true causal matrix
    if args.ground_truth:
        true_matrix = torch.tensor([
            [0, 0, 0, 0], 
            [1, 0, 0, 0], 
            [1, 0, 0, 0], 
            [0, 0, 1, 0]
        ]).to(args.device)

        if not os.path.exists(args.true_matrix):
            torch.save(true_matrix, args.true_matrix)
            logging.info(f"True matrix saved successfully at {args.true_matrix}.")
        else:
            logging.info(f"True matrix already exists at {args.true_matrix}, skipping save.")

    # Return the processed dataset
    return {
        "train_data": train_data,
        "val_data": val_data,
        "all_data": all_data,
        "input_dims": input_dims
    }




def athlete_1m(args, seed=42):
    """
    Loads and preprocesses Variant-MNIST image data for training.
    
    This function:
    1. Reads CSV files for image paths.
    2. Loads images and constructs feature tensors.
    3. Splits the dataset into training and validation sets.
    4. Downsamples images from 28x28 to 14x14.
    5. Flattens images and tabular data.
    6. Returns processed train and test datasets.
    
    Args:
        seed (int): Random seed for train/test split.
        data_source (str): Path to the dataset directory.

    Returns:
        dict: A dictionary containing train and test data.
    """


    X1, X21, X22, X23 = [], [], [], []
    Z21, Z22, Z23, Z24, Z25 = [], [], [], [], []
    X1 = np.load("./dataset/athlete_3m/X1_imagebind_embeddings.npy")
    X21 = np.load("./dataset/athlete_3m/X21.npy")
    X22 = np.load("./dataset/athlete_3m/X22.npy")
    X23 = np.load("./dataset/athlete_3m/X23.npy")
    Z21 = np.load("./dataset/athlete_3m/Z21.npy")
    Z22 = np.load("./dataset/athlete_3m/Z22.npy")
    Z23 = np.load("./dataset/athlete_3m/Z23.npy")
    Z24 = np.load("./dataset/athlete_3m/Z24.npy")
    Z25 = np.load("./dataset/athlete_3m/Z25.npy")
    
    # Dictionary to store original variables
    variables = {"X1": X1/255, "X21": X21, "X22": X22, "X23": X23, "Z21": Z21, "Z22": Z22, "Z23": Z23, "Z24": Z24, "Z25": Z25}
    train_val_split = {}
    for key, value in variables.items():
        train_data, val_data = train_test_split(value, test_size=0.1, random_state=seed)
        train_val_split[f"{key}_train"] = train_data
        train_val_split[f"{key}_val"] = val_data

    # Log train/validation set shapes
    logging.info("Training set shapes:")
    for key in ["X1_train", "X21_train", "X22_train", "X23_train", "Z21_train", "Z22_train", "Z23_train", "Z24_train", "Z25_train"]:
        logging.info(f"  {key}: {train_val_split[key].shape}")

    logging.info("Validation set shapes:")
    for key in ["X1_val", "X21_val", "X22_val", "X23_val", "Z21_val", "Z22_val", "Z23_val", "Z24_val", "Z25_val"]:
        logging.info(f"  {key}: {train_val_split[key].shape}")
        
    train_data = stack_data(train_val_split, "train", args.case_id)
    val_data = stack_data(train_val_split, "val", args.case_id)
    all_data = np.concatenate([train_data, val_data], axis=0)
    input_dims = [train_val_split["X1_train"].shape[1], train_val_split["X21_train"].shape[1]]
    
    logging.info(f"Final Train Data Shape: {train_data.shape}")
    logging.info(f"Final Test Data Shape: {val_data.shape}")
    logging.info(f"Final All Data Shape: {all_data.shape}")
    logging.info(f"Input dimensions are: {input_dims}")
    
    # Define the true causal matrix
    if args.ground_truth:
        true_matrix = torch.tensor([
            [0, 0, 0, 0], 
            [1, 0, 0, 0], 
            [1, 0, 0, 0], 
            [0, 0, 1, 0]
        ]).to(args.device)

        if not os.path.exists(args.true_matrix):
            torch.save(true_matrix, args.true_matrix)
            logging.info(f"True matrix saved successfully at {args.true_matrix}.")
        else:
            logging.info(f"True matrix already exists at {args.true_matrix}, skipping save.")

    # Return the processed dataset
    return {
        "train_data": train_data,
        "val_data": val_data,
        "all_data": all_data,
        "input_dims": input_dims
    }


