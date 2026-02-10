import logging
import os
import pickle
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


def setup_logging(log_path):
    """ Configure logging to output to both console and log file. """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),  # Save logs to file
            logging.StreamHandler()  # Print logs to console
        ]
    )

# Utility function to set random seed for reproducibility
def seed_everything(seed: int):
    """
    Set seed for all random number generators to ensure reproducibility.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_z_scatter_matrix(data, input_dims, args, mode="true"):
    """
    Plots a stylish scatter matrix for all latent variables (Z) in data,
    but only for the lower triangle of the matrix.

    Args:
        data (numpy.ndarray or torch.Tensor): The dataset containing X and Z variables.
        input_dims (list): The dimensions of the X variables to determine where Z starts.
        args (Namespace): Contains result_path for saving the figure.
        mode (str): "true" for true Z values, "est" for estimated Z values (torch.Tensor).

    Saves:
        A scatterplot matrix at args.result_path.
    """
    if mode == "true":
        # Identify the starting index of Z variables
        z_start_index = sum(input_dims)
        # Extract only the Z variables from data
        df_z = pd.DataFrame(
            data[:, z_start_index:], 
            columns=[f"Z{i+1}" for i in range(data.shape[1] - z_start_index)]
        )
        save_filename = "z_scatter_origin.png"

    elif mode == "est":
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()  # Convert tensor to numpy array
        df_z = pd.DataFrame(
            data, 
            columns=[f"Z{i+1}" for i in range(data.shape[1])]
        )
        save_filename = "z_scatter_est.png"

    else:
        raise ValueError("Invalid mode. Use 'true' or 'est'.")

    # Number of latent variables (Z)
    num_z = df_z.shape[1]

    # Create a stylish scatterplot matrix with lower triangle
    fig, axes = plt.subplots(num_z, num_z, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    # Use Seaborn styling
    sns.set_style("whitegrid")
    
    for i in range(num_z):
        for j in range(num_z):
            ax = axes[i, j]

            if i < j:
                # Upper triangle: Remove the axis
                ax.set_visible(False)
            elif i == j:
                # Diagonal: Show variable name
                ax.text(0.5, 0.5, df_z.columns[i], fontsize=14, fontweight='bold',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # Lower triangle: Scatter plot
                sns.scatterplot(x=df_z.iloc[:, j], y=df_z.iloc[:, i], ax=ax, alpha=0.6, s=10)
                ax.set_xlabel("")
                ax.set_ylabel("")

    # Set global X/Y label font size
    for ax in fig.get_axes():
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)

    # Save figure
    os.makedirs(args.result_path, exist_ok=True)
    save_path = os.path.join(args.result_path, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Scatterplot matrix saved at {save_path}")


def extract_graph_edges(text):
    lines = text.strip().split("\n")
    start_idx = lines.index("Graph Edges:") + 1  # +1 to skip the "Graph Edges:" line itself
    
    # Collect edges
    edges = []
    for line in lines[start_idx:]:
        if line.strip() == "":  # Stop at an empty line
            break
        edges.append(line.split('. ')[1])  # Skip the numbering (e.g., "1. ")
    
    return edges


def extract_graph_nodes(text):
    # Split the text into lines
    lines = text.strip().split("\n")
    
    # Find the line containing "Graph Nodes:"
    idx = lines.index("Graph Nodes:")
    
    # The next line contains the nodes separated by ';'
    nodes = lines[idx + 1].split(";")
    
    return nodes


def generate_adjacency_matrix_ordered(edges_list, nodes_order):
    """
    Generate an adjacency matrix based on the provided edges and node order.
    
    Parameters:
        edges_list (list): List of edges in the format "source relation target".
        nodes_order (list): List of node names in the desired order.
    
    Returns:
        numpy.ndarray: Adjacency matrix.
    """
    node_to_index = {node: idx for idx, node in enumerate(nodes_order)}
    
    # Initialize adjacency matrix with zeros
    adj_matrix = np.zeros((len(nodes_order), len(nodes_order)), dtype=int)
    
    # Update adjacency matrix based on edges
    for edge in edges_list:
        source, relation, target = edge.split(' ')
        if relation == '-->':
            adj_matrix[node_to_index[source], node_to_index[target]] = -1
            adj_matrix[node_to_index[target], node_to_index[source]] = 1
        elif relation == '<-->':
            adj_matrix[node_to_index[source], node_to_index[target]] = 1
            adj_matrix[node_to_index[target], node_to_index[source]] = 1
        elif relation == '---':
            adj_matrix[node_to_index[source], node_to_index[target]] = -1
            adj_matrix[node_to_index[target], node_to_index[source]] = -1

    return adj_matrix