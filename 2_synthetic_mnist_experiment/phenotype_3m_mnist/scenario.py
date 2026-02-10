import logging
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from utils.loader import load_image, downsample, flatten_data, stack_data


def phenotype(args, seed=42):
    csv_path = './dataset/phenotype/example.csv'
    output_files = {
        "m1": "./dataset/phenotype/m1.csv",
        "m2": "./dataset/phenotype/m2.csv",
        "m3": "./dataset/phenotype/m3.csv",
        "tabular": "./dataset/phenotype/tabular.csv"
    }
    
    # Load the main dataset
    df = pd.read_csv(csv_path)
    m1_cols = [col for col in df.columns if col.startswith("m1_")]
    m2_cols = [col for col in df.columns if col.startswith("m2_")]
    m3_cols = [col for col in df.columns if col.startswith("m3_")]
    tabular_cols = [col for col in df.columns if col not in set(m1_cols + m2_cols + m3_cols)]

    def load_or_generate_csv(df, filename, selected_columns):
        if os.path.exists(filename):
            logging.info(f"Loading existing file: {filename}")
            return pd.read_csv(filename)
        else:
            logging.info(f"Generating new file: {filename}")
            df_subset = df[selected_columns]
            df_subset.to_csv(filename, index=False)
            return df_subset

    df_m1 = load_or_generate_csv(df, output_files["m1"], m1_cols)
    df_m2 = load_or_generate_csv(df, output_files["m2"], m2_cols)
    df_m3 = load_or_generate_csv(df, output_files["m3"], m3_cols)
    df_tabular = load_or_generate_csv(df, output_files["tabular"], tabular_cols)

    X1 = df_m1[m1_cols].values
    X2 = df_m2[m2_cols].values
    X3 = df_m3[m3_cols].values

    # Log shapes
    logging.info(f"X1 (m1) shape: {X1.shape}")
    logging.info(f"X2 (m2) shape: {X2.shape}")
    logging.info(f"X3 (m3) shape: {X3.shape}")
    logging.info(f"Tabular shape: {df_tabular.shape}")

    all_data = np.concatenate([X1, X2, X3], axis=1)
    train_data = all_data
    val_data = all_data

    input_dims = [X1.shape[1], X2.shape[1], X3.shape[1]]

    logging.info(f"Final Train Data Shape: {train_data.shape}")
    logging.info(f"Final Test Data Shape: {val_data.shape}")
    logging.info(f"Final All Data Shape: {all_data.shape}")
    logging.info(f"Input dimensions are: {input_dims}")

    return {
        "train_data": train_data,
        "val_data": val_data,
        "all_data": all_data,
        "input_dims": input_dims
    }

    
def variant_mnist(args, seed=42):
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
    # Load CSV files containing metadata
    data_source = args.data_source
    df_color = pd.read_csv(os.path.join(data_source, "color.csv"))
    df_fashion = pd.read_csv(os.path.join(data_source, "fashion.csv"))

    # Initialize lists to store data
    X1, X2, Z11, Z12, Z21, Z22 = [], [], [], [], [], []

    # Load images and tabular data
    for (z11, z12, color_path), (z21, z22, fashion_path) in zip(df_color.values, df_fashion.values):
        X1.append(load_image(data_source, color_path))  # RGB image
        X2.append(load_image(data_source, fashion_path))  # Grayscale image
        Z11.append(z11)
        Z12.append(z12)
        Z21.append(z21) 
        Z22.append(z22)

    # Convert lists to NumPy arrays
    X1, X2 = np.array(X1), np.array(X2)
    Z11, Z12, Z21, Z22 = np.array(Z11).reshape(-1, 1), np.array(Z12).reshape(-1, 1), np.array(Z21).reshape(-1, 1), np.array(Z22).reshape(-1, 1)

    # Log dataset shapes
    logging.info(f"X1 shape: {X1.shape}, X2 shape: {X2.shape}")
    logging.info(f"Z11 shape: {Z11.shape}, Z12 shape: {Z12.shape}, Z21 shape: {Z21.shape}, Z22 shape: {Z22.shape}")

    # Dictionary to store original variables
    variables = {"X1": X1/255, "X2": X2/255, "Z11": Z11, "Z12": Z12, "Z21": Z21, "Z22": Z22}
    train_val_split = {}
    for key, value in variables.items():
        train_data, val_data = train_test_split(value, test_size=0.1, random_state=seed)
        train_val_split[f"{key}_train"] = train_data
        train_val_split[f"{key}_val"] = val_data

    # Log train/validation set shapes
    logging.info("Training set shapes:")
    for key in ["X1_train", "X2_train", "Z11_train", "Z12_train", "Z21_train", "Z22_train"]:
        logging.info(f"  {key}: {train_val_split[key].shape}")

    logging.info("Validation set shapes:")
    for key in ["X1_val", "X2_val", "Z11_val", "Z12_val", "Z21_val", "Z22_val"]:
        logging.info(f"  {key}: {train_val_split[key].shape}")

    # Downsample X1 (RGB) and X2 (Grayscale) in both training and validation sets
    train_val_split['X1_train'] = downsample(train_val_split['X1_train'], is_rgb=True)
    train_val_split['X2_train'] = downsample(train_val_split['X2_train'], is_rgb=False)
    train_val_split['X1_val'] = downsample(train_val_split['X1_val'], is_rgb=True)
    train_val_split['X2_val'] = downsample(train_val_split['X2_val'], is_rgb=False)

    # Log downsampled shapes
    logging.info("Downsampled training set shapes:")
    for key in ["X1_train", "X2_train"]:
        logging.info(f"  {key}: {train_val_split[key].shape}")

    logging.info("Downsampled validation set shapes:")
    for key in ["X1_val", "X2_val"]:
        logging.info(f"  {key}: {train_val_split[key].shape}")

    # Apply flattening
    train_val_split = flatten_data(train_val_split)
    
    # Log flattened data
    logging.info("Flattened data shapes:")
    for key in train_val_split.keys():
        logging.info(f"  {key}: {train_val_split[key].shape}")
        
    train_data = stack_data(train_val_split, "train")
    val_data = stack_data(train_val_split, "val")
    all_data = np.concatenate([train_data, val_data], axis=0)
    input_dims = [train_val_split["X1_train"].shape[1], train_val_split["X2_train"].shape[1]]
    
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


def variant_mnist_3m(args, seed=42):
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
    # Load CSV files containing metadata
    data_source = args.data_source
    df_color = pd.read_csv(os.path.join(data_source, "color.csv"))
    df_fashion = pd.read_csv(os.path.join(data_source, "fashion.csv"))

    # Initialize lists to store data
    X1, X2, Z11, Z12, Z21, Z22 = [], [], [], [], [], []
    X11, X12 = [], []

    # Load images and tabular data
    for (z11, z12, color_path, color_path_2, color_path_3), (z21, z22, fashion_path) in zip(df_color.values, df_fashion.values):
        X1.append(load_image(data_source, color_path))  # RGB image
        X11.append(load_image(data_source, color_path_2))  # RGB image
        X12.append(load_image(data_source, color_path_3))  # RGB image
        X2.append(load_image(data_source, fashion_path))  # Grayscale image
        Z11.append(z11)
        Z12.append(z12)
        Z21.append(z21) 
        Z22.append(z22)

    # Convert lists to NumPy arrays
    X1, X2 = np.array(X1), np.array(X2)
    X11, X12 = np.array(X11), np.array(X12) # green image & blue image
    Z11, Z12, Z21, Z22 = np.array(Z11).reshape(-1, 1), np.array(Z12).reshape(-1, 1), np.array(Z21).reshape(-1, 1), np.array(Z22).reshape(-1, 1)

    # Log dataset shapes
    logging.info(f"X1 shape: {X1.shape}, X2 shape: {X2.shape}")
    logging.info(f"Z11 shape: {Z11.shape}, Z12 shape: {Z12.shape}, Z21 shape: {Z21.shape}, Z22 shape: {Z22.shape}")
    logging.info(f"X11 shape: {X11.shape}, X12 shape: {X12.shape}")

    # Dictionary to store original variables
    variables = {"X1": X1/255, "X2": X2/255, "Z11": Z11, "Z12": Z12, "Z21": Z21, "Z22": Z22, "X11": X11/255, "X12": X12/255}
    train_val_split = {}
    for key, value in variables.items():
        train_data, val_data = train_test_split(value, test_size=0.1, random_state=seed)
        train_val_split[f"{key}_train"] = train_data
        train_val_split[f"{key}_val"] = val_data

    # Log train/validation set shapes
    logging.info("Training set shapes:")
    for key in ["X1_train", "X2_train", "Z11_train", "Z12_train", "Z21_train", "Z22_train"]:
        logging.info(f"  {key}: {train_val_split[key].shape}")

    logging.info("Validation set shapes:")
    for key in ["X1_val", "X2_val", "Z11_val", "Z12_val", "Z21_val", "Z22_val"]:
        logging.info(f"  {key}: {train_val_split[key].shape}")

    # Downsample X1 (RGB) and X2 (Grayscale) in both training and validation sets
    train_val_split['X1_train'] = downsample(train_val_split['X1_train'], is_rgb=True)
    train_val_split['X2_train'] = downsample(train_val_split['X2_train'], is_rgb=False)
    train_val_split['X1_val'] = downsample(train_val_split['X1_val'], is_rgb=True)
    train_val_split['X2_val'] = downsample(train_val_split['X2_val'], is_rgb=False)
    
    train_val_split['X11_train'] = downsample(train_val_split['X11_train'], is_rgb=True)
    train_val_split['X12_train'] = downsample(train_val_split['X12_train'], is_rgb=True)
    train_val_split['X11_val'] = downsample(train_val_split['X11_val'], is_rgb=True)
    train_val_split['X12_val'] = downsample(train_val_split['X12_val'], is_rgb=True)


    # Log downsampled shapes
    logging.info("Downsampled training set shapes:")
    for key in ["X1_train", "X2_train"]:
        logging.info(f"  {key}: {train_val_split[key].shape}")

    logging.info("Downsampled validation set shapes:")
    for key in ["X1_val", "X2_val"]:
        logging.info(f"  {key}: {train_val_split[key].shape}")

    # Apply flattening
    train_val_split = flatten_data(train_val_split)
    
    # Log flattened data
    logging.info("Flattened data shapes:")
    for key in train_val_split.keys():
        logging.info(f"  {key}: {train_val_split[key].shape}")
        
    train_data = stack_data(train_val_split, "train")
    val_data = stack_data(train_val_split, "val")
    all_data = np.concatenate([train_data, val_data], axis=0)
    input_dims = [train_val_split["X1_train"].shape[1], train_val_split["X11_train"].shape[1], train_val_split["X12_train"].shape[1], train_val_split["X2_train"].shape[1]]
    
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
