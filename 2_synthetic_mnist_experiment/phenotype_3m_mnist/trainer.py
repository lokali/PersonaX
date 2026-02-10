import torch
import numpy as np
import logging
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from models.causalmultimodal import CausalMultimodal
from models.causalphenotype import CausalPhenotye
from models.utils import calculate_r2, calculate_mcc, initialize_ground_truth_tracking


def evaluate_model_performance(epoch, model, input_dims, test_data, args, tracking):
    """
    Evaluates model performance using ground truth and updates tracking metrics.
    """
    model.eval()
    with torch.no_grad():
        # Extract test data inputs
        inputs_test = []
        start_idx = 0  # Initialize start index
        for dim in input_dims:
            end_idx = start_idx + dim
            input_tensor = torch.tensor(test_data[:, start_idx:end_idx], dtype=torch.float32).to(args.device)
            inputs_test.append(input_tensor)
            start_idx = end_idx  # Update start index for next slice
        
        # Forward pass through the model
        z_test, _, _, _, _ = model(args, epoch, inputs_test)

    # Correct extraction of ground truth latent variables
    # start_idx = 0  # Reset to correctly extract Z_true
    Z_true = np.hstack([test_data[:, start_idx + i:start_idx + i + 1] for i in range(sum(args.latent_dims))])
    Z_pred = z_test.detach().cpu().numpy()

    r2 = calculate_r2(Z_true, Z_pred)
    mcc = calculate_mcc(Z_true, Z_pred)
    tracking["r2_list"].append(r2)
    tracking["mcc_list"].append(mcc)
    tracking["epochs_recorded"].append(epoch + 1)

    if tracking["best_mcc"] is None or mcc > tracking["best_mcc"]:
        tracking["best_mcc"], tracking["best_r2"], tracking["best_epoch"] = mcc, r2, epoch + 1

    logging.info(
        f'Epoch [{epoch+1}/{args.num_epochs}], R2: {r2:.4f}, MCC: {mcc:.4f}, '
        f'Total Loss: {tracking["total_loss"]:.4f}, '
        f'Recon: {tracking["recon_loss"]:.4f}, '
        f'KLD: {tracking["kld_loss"]:.4f}, '
        f'Sparsity: {tracking["sparsity_loss"]:.4f}, '
        f'LR: {tracking["current_lr"]}'
    )

def train_model(args, input_dims, train_data, test_data):
    # Initialize model
    if args.case_id == 'phenotype':
        model = CausalPhenotye(
            n_modalities = args.n_modalities,
            input_dims = input_dims,
            input_types = args.input_types,
            latent_dims = args.latent_dims,
            eta_dims = args.eta_dims,
            hidden_dims = args.hidden_dims,
            shared_dims = args.shared_dims,
            alpha_threshold = args.alpha_threshold, 
            result_path = args.result_path,
            enable_shared_latent = args.enable_shared_latent,
            enable_inverse_gaussian = args.enable_inverse_gaussian
        ).to(args.device)
    else:
        model = CausalMultimodal(
            n_modalities = args.n_modalities,
            input_dims = input_dims,
            input_types = args.input_types,
            latent_dims = args.latent_dims,
            eta_dims = args.eta_dims,
            hidden_dims = args.hidden_dims,
            shared_dims = args.shared_dims,
            alpha_threshold = args.alpha_threshold, 
            result_path = args.result_path,
            enable_shared_latent = args.enable_shared_latent,
            enable_inverse_gaussian = args.enable_inverse_gaussian
        ).to(args.device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    
    tracking = initialize_ground_truth_tracking() if args.ground_truth else {}
    
    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        inputs_train = [
            torch.tensor(train_data[:, start:start + dim], dtype=torch.float32).to(args.device)
            for start, dim in zip(np.cumsum([0] + input_dims[:-1]), input_dims)
        ]
        _, _, recon_loss, kld_loss, sparsity_loss = model(args, None, inputs_train)

        tracking["total_loss"] = (
            args.recon_coef * recon_loss +
            args.kl_coef * kld_loss +
            args.sparsity_coef * sparsity_loss
        )
        tracking["recon_loss"] = recon_loss
        tracking["kld_loss"] = kld_loss
        tracking["sparsity_loss"] = sparsity_loss

        optimizer.zero_grad()
        tracking["total_loss"].backward()
        optimizer.step()
        scheduler.step()
        tracking["current_lr"] = scheduler.get_last_lr()[0]

        if epoch % args.est_itr == 0 and args.ground_truth:
            evaluate_model_performance(epoch, model, input_dims, test_data, args, tracking)

        if epoch % args.est_itr == 0 and not args.ground_truth:
            logging.info(
                f'Epoch [{epoch+1}/{args.num_epochs}], '
                f'Total Loss: {tracking["total_loss"]:.4f}, '
                f'Recon: {tracking["recon_loss"]:.4f}, '
                f'KLD: {tracking["kld_loss"]:.4f}, '
                f'Sparsity: {tracking["sparsity_loss"]:.4f}, '
                f'LR: {tracking["current_lr"]}'
            )

    if args.ground_truth:
        model._save_results(tracking["epochs_recorded"], tracking["r2_list"], tracking["mcc_list"])
        logging.info(f"Highest MCC: {tracking['best_mcc']:.4f} at epoch {tracking['best_epoch']} with R2: {tracking['best_r2']:.4f}")

    return model
