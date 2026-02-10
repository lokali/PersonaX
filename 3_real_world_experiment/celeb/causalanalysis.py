import io
import pydot
import pickle
import logging
import subprocess
from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
from phenofeature import category_to_features


def visualize_causal_graph(graph, labels_all, save_path, causal_order=None):
    """
    Converts a causal graph to a PNG visualization with feature-based coloring.

    Args:
        graph: Causal graph object from causal discovery.
        labels_all: List of feature names corresponding to graph nodes.
        save_path: Path to save the output PNG file.
        causal_order: Optional list of causal layer order for visualization.
    """
    # Convert causal graph to pydot format
    pyd = GraphUtils.to_pydot(graph.G, labels=labels_all)
    if not pyd or not isinstance(pyd, pydot.Dot):
        raise ValueError("❌ Error: `pyd` is not a valid pydot.Dot object!")

    dot_code = pyd.to_string()
    if not dot_code.strip():
        raise ValueError("❌ Error: The DOT content is empty!")

    # Configure graph aesthetics
    pyd.set_rankdir("LR")  # Left to right layout
    pyd.set_dpi(300)
    pyd.set_nodesep(0.15)
    pyd.set_ranksep(0.2)
    pyd.set_splines("polyline")
    
    # Save the DOT file
    dot_path = save_path.rsplit(".", 1)[0] + ".dot"
    pyd.write(dot_path, format="dot")  # Corrected format

    # Convert .dot to .png using Graphviz CLI (more reliable than pydot create_png)
    try:
        subprocess.run(["dot", "-Tpng", dot_path, "-o", save_path], check=True)
        print(f"✅ Causal graph saved at: {save_path}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"❌ Error generating PNG: {e}")

    # Load and display the PNG
    img = mpimg.imread(save_path)
    plt.figure(figsize=(10, 8))
    plt.axis("off")
    plt.imshow(img)
    plt.savefig(save_path, bbox_inches="tight")


def perform_causal_discovery(args, updated_data_path, selected_keys):
    """
    Perform causal analysis on the updated dataset and save the results.
    """
    # Load the updated dataset (CSV or Excel)
    if updated_data_path.endswith('.csv'):
        data = pd.read_csv(updated_data_path)
    elif updated_data_path.endswith('.xlsx'):
        data = pd.read_excel(updated_data_path)
    else:
        raise ValueError("Unsupported file format for causal analysis.")

    logging.info(f"The dataset contains {data.shape[1]} features.")

    # Extract selected keys from the dataset
    if not selected_keys:
        selected_keys = list(data.columns)
        logging.info("No selected_keys provided. Using all columns for causal analysis.")

    # Perform causal discovery using the PC algorithm
    df = data[selected_keys]
    logging.info(f"The causal analyzed dataset contains {df.shape[1]} features.")
    
    labels = df.columns
    cg = pc(df.to_numpy(), alpha=args.pvalue, indep_test=args.indep_test, stable=True, verbose=False)

    save_path = f'{args.result_path}/{args.causal_method}_{args.pvalue}_{args.indep_test}.png'
    visualize_causal_graph(cg, labels, save_path)

    logging.info(f"Causal graph saved: {save_path}")
