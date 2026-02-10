
import os
import re
import numpy as np
import argparse
from glob import glob

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

def run_pc(data, alpha, indep_test, label, file_id, epoch, output_prefix):
    """
    Runs the PC algorithm on the given data and saves the resulting graph.

    Parameters:
    - data (np.ndarray): The dataset to process.
    - alpha (float): Significance level for independence tests.
    - indep_test (str): Name of the independence test.
    - label (list[str]): List of variable names.
    - file_id (str): ID extracted from directory.
    - epoch (str): Epoch number extracted from filename.
    - output_prefix (str): Path prefix to save the results.
    """
    print(f"[RUNNING] {output_prefix} - Shape: {data.shape}")
    bk = BackgroundKnowledge()
    

    # fisher
    cg0 = pc(data, alpha, indep_test='fisherz', verbose=False, show_progress=True)
    nodes = cg0.G.get_nodes()
    
    for i in range(3, 18):
        print(nodes[i])
        bk.add_forbidden_by_node(nodes[i], nodes[0])
        bk.add_forbidden_by_node(nodes[i], nodes[1])
        bk.add_forbidden_by_node(nodes[i], nodes[2])
    # bk.add_required_by_node(nodes[labels_all.index('IFNg_VS1')], nodes[labels_all.index('TNFa_VS1')])


    cg = pc(data, alpha, indep_test, background_knowledge=bk)

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    # Save .npy
    np.save(output_prefix + ".npy", cg.G.graph)

    # Save .png
    pyd = GraphUtils.to_pydot(cg.G, labels=label)
    pyd.write_png(output_prefix + ".png")

def extract_file_id_and_epoch(filepath):
    file_id = os.path.basename(os.path.dirname(filepath))
    match = re.search(r'epoch(\d+)', os.path.basename(filepath))
    epoch = match.group(1) if match else "unknown"
    return file_id, epoch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch run PC algorithm on .npy files in a directory.")
    parser.add_argument("input_dir", type=str, help="Path to directory containing .npy files")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--indep_test", type=str, default="rcit", help="Independence test (e.g. kci, fisherz, rcit)")
    parser.add_argument("--label_count", type=int, default=18, help="Number of labels (default: 12)")
    parser.add_argument("--output_dir", type=str, default="results/pc_rcit_wforbidden_cel", help="Directory to save results")

    args = parser.parse_args()
    np.random.seed(0)

    # labels = [f"x{i}" for i in range(12)] + ['final_o', 'final_c', 'final_e', 'final_a', 'final_n', 'Birthyear', 'Birthmonth', 'Birthday', 'Type', 'Latitude', 'Longitude', 'Height', 'Weight']

    npy_files = glob(os.path.join(args.input_dir, "**", "*.npy"), recursive=True)

    print(f"Found {len(npy_files)} .npy files to process.")

    for npy_file in npy_files:
        
        if "pred" not in os.path.basename(npy_file):
            continue

        try:
            data = np.load(npy_file, allow_pickle=True)
            file_id, epoch = extract_file_id_and_epoch(npy_file)

            labels = [f"x{i}" for i in range(18)]

            alpha_str = str(args.alpha).replace(".", "_")
            base_filename = f"{file_id}_epoch{epoch}_{args.indep_test}_alpha{alpha_str}_D{data.shape[1]}_N{data.shape[0]}"
            output_path = os.path.join(args.output_dir, base_filename)

            run_pc(data, args.alpha, args.indep_test, labels, file_id, epoch, output_path)

        except Exception as e:
            print(f"[ERROR] Skipped {npy_file}: {e}")
