
import os
import numpy as np

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils

def run_pc(data, alpha, indep_test, label):
    """
    Runs the PC algorithm on the given data and saves the resulting graph.

    Parameters:
    - data (numpy.ndarray or pd.DataFrame.values): The dataset to process.
    - alpha (float): Significance level for independence tests.
    - indep_test (str): Name of the independence test ('kci', 'fisherz', etc.).

    Returns:
    - cg.G (causallearn graph object): The learned causal graph.
    """
    print(f"Running PC algorithm on data with shape: {data.shape}")

    # Run the PC algorithm
    cg = pc(data, alpha, indep_test)

    # Ensure the results directory exists
    dir = './dataset/athlete_3m/results/'
    if not os.path.exists(dir):
        os.mkdir(dir)

    # Sanitize filename (replace '.' in alpha with '_')
    alpha_str = str(alpha).replace(".", "_")
    filename = f"{indep_test}_alpha{alpha_str}_D{data.shape[1]}_N{data.shape[0]}"
    file_path = os.path.join(dir, filename)

    # Save the graph
    np.save(file_path + '.npy', cg.G.graph)

    # save plain graph
    cg.draw_pydot_graph(label) # show on display
    pyd = GraphUtils.to_pydot(cg.G, labels=label)
    pyd.write_png(file_path + '.png')
    return cg, file_path


labels = [f"x{i}" for i in range(12)]
graph_array = np.load('', allow_pickle=True)
run_pc(graph_array, 0.5, 'kci', labels)

