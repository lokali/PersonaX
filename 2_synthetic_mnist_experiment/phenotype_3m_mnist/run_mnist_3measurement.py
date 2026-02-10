import os
import random
import argparse
import torch
import logging
from utils.utils import seed_everything, setup_logging
from scenario import variant_mnist, variant_mnist_3m
from trainer import train_model
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.debug=false"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="Run causal representation learning from multimodal pipeline.")
    
    # Experiment Settings
    parser.add_argument('--exp_choice', type=str, default='mnist', choices=['synthetic', 'mnist', 'phenotype'])
    parser.add_argument('--case_id', type=str, default='variant_mnist_3m')
    parser.add_argument('--seed', type=int, default=random.randint(0, 2**32 - 1))
    parser.add_argument('--loaded_data_path', type=str, default=None, 
                        help="Path to an existing dataset (CSV or Excel). If not provided, only the representation will be used.")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        choices=['cpu', 'cuda', 'mps'], help="Device to use for training (cpu, cuda, mps)")
    
    # Data Configuration
    parser.add_argument('--n_modalities', type=int, default=2)
    parser.add_argument('--eta_dims', type=int, nargs='+', default=[1, 1, 1, 1])
    parser.add_argument('--latent_dims', type=int, nargs='+', default=[2, 2, 2, 2])
    parser.add_argument('--input_types', type=str, nargs='+', default=['mnist', 'mnist', 'mnist', 'mnist'])
    parser.add_argument('--input_order', type=str, nargs='+', default=['mnist', 'mnist', 'mnist', 'mnist'])
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[8, 8, 8, 8])
    parser.add_argument('--shared_dims', type=int, nargs='+', default=[1, 1, 1, 1])
    
    # Training Parameters
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--num_epochs', type=int, default=int(3e3))
    parser.add_argument('--scheduler_step', type=int, default=int(1e4))
    parser.add_argument('--scheduler_gamma', type=float, default=0.99)
    parser.add_argument('--recon_coef', type=float, default=2)
    parser.add_argument('--kl_coef', type=float, default=1e-2)
    parser.add_argument('--sparsity_coef', type=float, default=1e-3)
    parser.add_argument('--alpha_threshold', type=float, default=1e-2)
    parser.add_argument('--est_itr', type=int, default=100)
    parser.add_argument('--plot_itr', type=int, default=1000)
    parser.add_argument('--enable_shared_latent', action='store_true')
    parser.add_argument('--enable_inverse_gaussian', action='store_true')
    parser.add_argument("--ground_truth", action="store_true", help="Set to True if ground truth is available.")

    # Causal Discovery
    parser.add_argument('--causal_method', type=str, default="pc")
    parser.add_argument('--pvalue', type=float, default=0.1)
    parser.add_argument('--indep_test', type=str, default="fisherz")

    return parser.parse_args()


def main():
    args = parse_args()
    args.seed = 687590513
    seed_everything(args.seed)
    
    DATA_SCENARIOS = {
        "variant_mnist_3m": lambda: variant_mnist_3m(args) 
    }
    if args.case_id not in DATA_SCENARIOS:
        raise ValueError(
            f"Error: `args.case_id` should be one of {list(DATA_SCENARIOS.keys())}, but got '{args.case_id}'."
        )
    
    args.data_source = f'./dataset/{args.case_id}'
    args.result_path = f'./result/{args.case_id}/{args.seed}'
    args.true_matrix = f'{args.result_path}/true_matrix.pt'
    args.log_file_path = f'{args.result_path}/experiment.log'
    
    os.makedirs(args.result_path, exist_ok=True)
    os.makedirs(args.data_source, exist_ok=True)

    setup_logging(args.log_file_path)
    logging.info("All arguments passed to the script:")
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")
    logging.info("=" * 50)
    logging.info(f"Experiment case ID: {args.case_id}")
    logging.info(f"Random seed used: {args.seed}")

    # Step 1: Load the data
    logging.info(f"Starting loading the data from {args.case_id}...")
    if args.case_id in DATA_SCENARIOS:
        dataset = DATA_SCENARIOS[args.case_id]() 
        logging.info(f"Successfully loaded dataset for {args.case_id}.")
    else:
        logging.error(f"Error: Unknown case_id '{args.case_id}'. Available options: {list(DATA_SCENARIOS.keys())}")
        return 
    train_data, val_data, all_data, input_dims = dataset["train_data"], dataset["val_data"], dataset["all_data"], dataset["input_dims"]

    # Step 2: Run pipeline
    logging.info("Starting model training...")
    model = train_model(args, input_dims, train_data, val_data)

    logging.info("Pipeline execution complete. Results saved in: " + args.result_path)


if __name__ == "__main__":
    main()
