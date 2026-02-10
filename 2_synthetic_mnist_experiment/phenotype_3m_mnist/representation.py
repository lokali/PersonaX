import os
import torch
import numpy as np
import pandas as pd
import logging

def update_representation(args, representation):
    """
    Update the existing dataset (if available) with new representations and save.
    - If an existing dataset is found, merge it with the new representation.
    - If no dataset exists, save only the representation.
    """
    # Ensure `input_types` and `latent_dims` match
    if len(args.input_types) != len(args.latent_dims):
        raise ValueError("The number of `input_types` must match the length of `latent_dims`.")

    # Generate column names dynamically based on input_types and latent_dims
    rep_columns = []
    col_index = 0
    for input_type, dim in zip(args.input_types, args.latent_dims):
        rep_columns.extend([f"{input_type}_{i+1}" for i in range(col_index, col_index + dim)])
        col_index += dim  # Update column index for next input_type

    # Convert representation tensor to DataFrame
    rep_df = pd.DataFrame(representation.cpu().numpy(), columns=rep_columns)

    if args.loaded_data_path and os.path.exists(args.loaded_data_path):
        logging.info(f"Existing data found: {args.loaded_data_path}")
        
        if args.loaded_data_path.endswith('.csv'):
            existing_data = pd.read_csv(args.loaded_data_path)
        elif args.loaded_data_path.endswith(('.xls', '.xlsx')):
            existing_data = pd.read_excel(args.loaded_data_path)
        else:
            raise ValueError("Unsupported file format. Only CSV and Excel are supported.")
        
        # Merge the existing dataset with the new representation
        updated_data = pd.concat([existing_data, rep_df], axis=1)
    else:
        logging.info("No existing data found. Saving representation only.")
        updated_data = rep_df

    output_file = os.path.join(args.result_path, f"{args.case_id}_combined.csv")
    
    if args.loaded_data_path and args.loaded_data_path.endswith('.xlsx'):
        output_file = output_file.replace('.csv', '.xlsx')
        updated_data.to_excel(output_file, index=False)
    else:
        updated_data.to_csv(output_file, index=False)

    logging.info(f"Updated data saved to {output_file}")
    
    return output_file


def extract_representation(model, all_data, input_dims, args):
    """
    Extract the learned representation from the model.
    
    Args:
        model (torch.nn.Module): The trained model.
        all_data (np.ndarray or torch.Tensor): Input data for inference.
        input_dims (list): List of input dimensions for each modality.
        args (argparse.Namespace): Arguments containing device and file paths.

    Returns:
        torch.Tensor: Extracted representation.
    """
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient tracking
        # Convert all_data into modality-specific tensors
        inputs = [
            torch.tensor(all_data[:, start:start + dim], dtype=torch.float32).to(args.device)
            for start, dim in zip(np.cumsum([0] + input_dims[:-1]), input_dims)
        ]
        
        # Get representation from the model
        representation, _, _, _, _ = model(args, None, inputs)

    logging.info(f"The shape of learned representation is {representation.shape}")

    # Ensure `input_types` and `latent_dims` match
    if len(args.input_types) != len(args.latent_dims):
        raise ValueError("The number of `input_types` must match the length of `latent_dims`.")

    # Generate column names dynamically based on input_types and latent_dims
    rep_columns = []
    col_index = 0
    for input_type, dim in zip(args.input_types, args.latent_dims):
        rep_columns.extend([f"{input_type}_{i+1}" for i in range(col_index, col_index + dim)])
        col_index += dim  # Update column index for next input_type
    representation_df = pd.DataFrame(representation.cpu().numpy(), columns=rep_columns)
    representation_df.to_csv(f"{args.result_path}/{args.case_id}_representation.csv", index=False)

    logging.info(f"Representation saved to {args.result_path}/{args.case_id}_representation.csv")

    return representation