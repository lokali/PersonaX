import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np 
from typing import List, Tuple
from cdt.metrics import SHD
import matplotlib.pyplot as plt


def initialize_weights(module):
    """Initialize weights using Kaiming normal initialization."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class PerDimMLP(nn.Module):
    """ Computes e_i, where the number of parent nodes can vary for each input """
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output e_i
        )

    def forward(self, z_i, z_parent):
        """
        Args:
        z_i: (batch, 1) - The current variable
        z_parent: (batch, num_parents) - The parent variables (number of parents varies)

        Returns:
        e_i: (batch, 1) - The computed e_i value
        """
        if z_parent.shape[1] == 0:
            z_parent_agg = torch.zeros_like(z_i)  # If no parents exist, fill with 0
        else:
            z_parent_agg = z_parent.mean(dim=1, keepdim=True)  # Aggregate parents by mean

        mlp_input = torch.cat([z_i, z_parent_agg], dim=-1)  # (batch, 2)
        return self.mlp(mlp_input)


class EstimateE(nn.Module):
    """ Estimates E using an MLP for each dimension """
    def __init__(self, input_dim, hidden_dim=4):
        super().__init__()
        self.mlp_list = nn.ModuleList([PerDimMLP(hidden_dim) for _ in range(input_dim)])

    def forward(self, args, Z, matrix):
        """
        Args:
        args: Object containing hyperparameters
        Z: (batch_size, dim) - Input variables
        matrix: (dim, dim) - Adjacency matrix representing parent-child relationships

        Returns:
        E: (batch_size, dim) - Estimated E values
        """
        batch_size, dim = Z.shape
        alpha_est = matrix * (matrix > args.alpha_threshold).float()  # Apply thresholding
        E_list = []
        for i in range(dim):
            mask = alpha_est[i] > args.alpha_threshold  # Identify parent indices
            Z_parent_i = Z[:, mask]  # Select parent variables

            e_i = self.mlp_list[i](Z[:, i:i+1], Z_parent_i)  # Compute e_i
            E_list.append(e_i)

        # Concatenate all e_i values to form E
        E = torch.cat(E_list, dim=-1)  # (batch_size, dim)
        return E


# Define the inverse Gaussian transformation
def inverse_gaussian(data):
    """
    Converts data to ranks, normalizes ranks to quantiles, and applies the inverse Gaussian CDF.
    The operation is performed on PyTorch tensors, supporting GPU and differentiability.
    """
    # Get ranks: torch.argsort twice gives ranks (equivalent to NumPy's np.argsort(np.argsort))
    ranks = torch.argsort(torch.argsort(data, dim=0), dim=0).float() + 1  # Rank starts from 1
    n = data.shape[0]
    # Normalize ranks to quantiles
    quantiles = (ranks - 0.5) / n
    # Apply the inverse CDF of a standard normal distribution (Inverse Gaussian transform)
    transformed_data = D.Normal(0, 1).icdf(quantiles)  # PyTorch equivalent of scipy.stats.norm.ppf
    return transformed_data


def calculate_kl_divergence(eps: torch.Tensor, exo: torch.Tensor) -> torch.Tensor:
    """Calculate the KL divergence for latent variables."""
    E = torch.cat([eps, exo], dim=-1)
    batch_size, dim = E.size()
    I = torch.eye(dim).to(E.device)

    empirical_mean = torch.mean(E, dim=0, keepdim=True)
    empirical_cov = torch.matmul((E - empirical_mean).T, (E - empirical_mean)) / batch_size

    _, empirical_cov_det = torch.slogdet(empirical_cov)
    kl_loss = -empirical_cov_det - dim + torch.trace(empirical_cov) + torch.sum(empirical_mean ** 2)

    return kl_loss


def flow_kl_divergence(args, eps, eta, log_det):
    """
    Computes the negative log-likelihood of a joint Gaussian distribution.
    
    Args:
        eps (Tensor): The vector output from the model.
        eta (Tensor): An additional vector.
        log_det (Tensor): Log-determinant of the transformation.
    
    Returns:
        Tensor: The negative log-likelihood.
    """
    joint_vector = torch.cat([eps, eta], dim=1)
    if args.enable_inverse_gaussian:
        joint_vector = inverse_gaussian(joint_vector)
    prior_log_prob = -0.5 * torch.sum(joint_vector ** 2, dim=1) - 0.5 * np.log(2 * np.pi) * joint_vector.size(1)
    log_prob = prior_log_prob + log_det  
    return -torch.mean(log_prob)


class tabularEncoder(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        latent_dim: int, 
        eta_dim: int,
        hidden_dim: int
        ):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim + eta_dim)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = F.leaky_relu(self.fc1(x))
        hidden = F.leaky_relu(self.fc2(hidden))
        output = self.fc3(hidden)
        z, eta = output[:, :self.latent_dim], output[:, self.latent_dim:]
        return z, eta


class tabularDecoder(nn.Module):
    def __init__(
        self, 
        latent_dim: int, 
        output_dim: int,
        hidden_dim: int
        ):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
            
    def forward(self, z: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        x_input = torch.cat([z, eta], dim=-1)
        hidden = F.leaky_relu(self.fc1(x_input))
        hidden = F.leaky_relu(self.fc2(hidden))
        x_recon = self.fc3(hidden)
        return x_recon


class mnistEncoder(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        latent_dim: int, 
        eta_dim: int,
        hidden_dim: int
        ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, latent_dim + eta_dim)

        self.latent_dim = latent_dim
        self.eta_dim = eta_dim
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        hidden = F.elu(self.fc1(x))
        hidden = F.elu(self.fc2(hidden))
        hidden = F.elu(self.fc3(hidden))
        output = self.fc4(hidden)
        # first output is z, then eta
        z, eta = output[:, :self.latent_dim], output[:, self.latent_dim:]
        return z, eta


class mnistDecoder(nn.Module):
    def __init__(
        self, 
        latent_dim: int, 
        output_dim: int,
        hidden_dim: int
        ):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 128)   
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, output_dim)
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, z, eta):
        x_input = torch.cat([z, eta], dim=-1)
        hidden = F.elu(self.fc1(x_input))
        hidden = F.elu(self.fc2(hidden))
        hidden = F.elu(self.fc3(hidden))
        x_recon = self.fc4(hidden)
        return x_recon


class UpdateE(nn.Module):
    def __init__(
        self, input_dim: int = 1, hidden_dim: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        return self.fc(E)


class RealNVP(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers):
        super(RealNVP, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim // 2 * 2)
            ) for _ in range(num_layers)
        ])

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, z):
        log_det = 0
        for i in range(self.num_layers):
            z1, z2 = z.chunk(2, dim=1)
            
            h = self.transforms[i](z1)
            shift, scale = h.chunk(2, dim=1)
            scale = torch.sigmoid(scale + 2)  
            
            z2 = z2 * scale + shift
            z = torch.cat([z1, z2], dim=1)
            
            log_det += torch.sum(torch.log(scale), dim=1)
        
        return z, log_det

    def inverse(self, eps):
        for i in reversed(range(self.num_layers)):
            z1, z2 = eps.chunk(2, dim=1)
            
            h = self.transforms[i](z1)
            shift, scale = h.chunk(2, dim=1)
            scale = torch.sigmoid(scale + 2) 
            
            z2 = (z2 - shift) / scale
            eps = torch.cat([z1, z2], dim=1)
        
        return eps


class CausalMultimodal(nn.Module):
    def __init__(
        self, 
        n_modalities: int, 
        input_dims: List[int], 
        input_types: List[str],
        latent_dims: List[int],
        eta_dims: List[int], 
        hidden_dims: List[int],
        shared_dims: List[int],
        alpha_threshold: float, 
        result_path: str,
        enable_shared_latent: bool = False,
        enable_inverse_gaussian: bool = False
        ):
        """
        This network supports multiple input modalities, each with a custom encoder-decoder structure.

        Args:
            n_modalities (int): Maximum number of modalities for forward pass.
            input_dims (List[int]): Dimensions of each input modality.
            input_types (List[str]): Types of input modalities (e.g., "fundus", "sleep", "bone_density").
            latent_dims (List[int]): Dimensions of the modality-specific latent space.
            eta_dims (List[int]): Dimensions of eta in each modality.
            hidden_dims (List[int]): Dimensions of the hidden neuros in each encoder/decoder.
            shared_dims (List[int]): Dimensions of the shared latent space among modalities.
            alpha_threshold (float): Sparsity threshold for the causal graph
            enable_shared_latent (bool): Whether or not allow the shared latent among modalities. Defaults to False.
            enable_inverse_gaussian (bool): Whether or not to use inverse Gaussian for linearization. Defaults to False.
        """
        super().__init__()
        # assert len(input_dims) == n_modalities, "Error: `input_dims` length must match `n_modalities`."
        # assert len(input_types) == n_modalities, "Error: `input_types` length must match `n_modalities`."
        # assert len(latent_dims) == n_modalities, "Error: `latent_dims` length must match `n_modalities`."
        # assert len(eta_dims) == n_modalities, "Error: `eta_dims` length must match `n_modalities`."
        # assert len(hidden_dims) == n_modalities, "Error: `hidden_dims` length must match `n_modalities`."
        # assert len(shared_dims) == n_modalities, "Error: `shared_dims` length must match `n_modalities`."

        self.n_modalities = n_modalities
        self.l_dim = latent_dims[0]
        self.eta_dims = eta_dims
        self.input_dims = input_dims
        self.input_types = input_types
        self.latent_dims = latent_dims
        self.shared_dims = shared_dims
        self.alpha_threshold = alpha_threshold
        self.enable_shared_latent = enable_shared_latent
        self.enable_inverse_gaussian = enable_inverse_gaussian
        self.result_path = result_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # constraint for sparsity in the causal graph
        self.num_latent = latent_dims[0] + latent_dims[1] + 2
        self.A = nn.Parameter((torch.randn(self.num_latent, self.num_latent) * 1e-5) + 0.5)
        self.A.data.fill_diagonal_(0)
        
        self.b = nn.Parameter(torch.randn(self.num_latent))
        
        # constraint for epsion in the causal process
        # self.updatee = nn.ModuleList([UpdateE() for _ in range(self.num_latent)])
        self.updatee = RealNVP(dim=self.num_latent, hidden_dim=16, num_layers=5)
        self.transfer = EstimateE(input_dim=self.num_latent)

        # print(f"the input types: {input_types} | the input dims: {input_dims} | the latent dims: {latent_dims} | the eta dims: {eta_dims} | the hidden dims: {hidden_dims}")
        
        # different encoder & decoder structure for different modality
        self.encoders = nn.ModuleList([
            self._select_encoder(input_type, input_dim, latent_dim, eta_dim, hidden_dim)
            for input_type, input_dim, latent_dim, eta_dim, hidden_dim in zip(input_types, input_dims, latent_dims, eta_dims, hidden_dims)
        ])
        self.decoders = nn.ModuleList([
            self._select_decoder(input_type, latent_dim, eta_dim, input_dim, hidden_dim)
            for input_type, latent_dim, input_dim, eta_dim, hidden_dim in zip(input_types, latent_dims, input_dims, eta_dims, hidden_dims)
        ])

        self.encoder_for_shared = self._select_encoder('mnist', np.sum(input_dims), 2, 1, None)
        self.decoder_for_shared = self._select_decoder('mnist', 2, 1, np.sum(input_dims), None)


    def _save_results(self, epochs, r2_list, mcc_list):
        np.savez(f'{self.result_path}/metrics_result', epochs=epochs, r2=r2_list, mcc=mcc_list)
        plt.style.use("seaborn-v0_8-darkgrid")
        plt.figure(figsize=(8, 6)) 
        plt.plot(epochs, r2_list, label='R² Score', color='blue', marker='o', linestyle='-', linewidth=2, markersize=6)
        plt.plot(epochs, mcc_list, label='MCC', color='red', marker='s', linestyle='--', linewidth=2, markersize=6)
        plt.xlabel('Epoch', fontsize=14, fontweight='bold')
        plt.ylabel('Metrics', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(f'{self.result_path}/metrics_plot.png', dpi=300, bbox_inches='tight')
        plt.close()


    def _select_encoder(self, input_type: str, input_dim: int, latent_dim: int, eta_dim: int, hidden_dim: int):
        """Select appropriate encoder based on the input type."""
        if input_type == "mnist":
            return mnistEncoder(input_dim, latent_dim, eta_dim, hidden_dim)
        elif input_type == "synthetic":
            return tabularEncoder(input_dim, latent_dim, eta_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")


    def _select_decoder(self, input_type: str, latent_dim: int, eta_dim: int, output_dim: int, hidden_dim: int):
        """Select appropriate decoder based on the input type."""
        if input_type == "mnist":
            return mnistDecoder(latent_dim + eta_dim, output_dim, hidden_dim)
        elif input_type == "synthetic":
            return tabularDecoder(latent_dim + eta_dim, output_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")


    def forward(self, args, epoch, inputs: List[torch.Tensor]):
        ##################################
        # Step 1: Encoders for all inputs
        ##################################
        z_group = []
        eps_group = []
        eta_group = []
        z_all = []
        num_inputs = len(inputs)


        ##################################
        # Step 1.1: Encoder & Decoder for the shared latents 
        ##################################
        all_inputs = torch.cat(inputs, dim=1)
        z_shared, eta_shared = self.encoder_for_shared(all_inputs)
        # print(f"z_shared: {z_shared.shape} | eta_shared: {eta_shared.shape}")
        z_all.append(z_shared)
        # eta_group.append(eta_shared)
        recon_x = self.decoder_for_shared(z_shared, eta_shared)
        recon_loss = F.mse_loss(recon_x, all_inputs)
        reconstruction_loss = recon_loss


        for i in range(num_inputs):
            current_x = inputs[i]
            current_z, current_eta = self.encoders[i](current_x)
            z_group.append(current_z)
            eta_group.append(current_eta)
            if i == 0 or i == 1:
                z_all.append(current_z)
        z_all = torch.cat(z_all, dim=1)
        eta_all = torch.cat(eta_group, dim=1)
        z_gap_loss = (torch.norm(z_group[1] - z_group[2], p=2) + torch.norm(z_group[1] - z_group[3], p=2)) / 2
            
        ##################################
        # Step 2: Decoders for all groups
        ##################################
        x_all = []
        x_hat_all = []
        # reconstruction_loss = 0
        for i in range(num_inputs):
            current_x = inputs[i]
            recon_x = self.decoders[i](z_group[i], eta_group[i])
            recon_loss = F.mse_loss(recon_x, current_x)
            reconstruction_loss += recon_loss
            x_all.append(current_x)
            x_hat_all.append(recon_x)
        x_all = torch.cat(x_all, dim=1)
        x_hat_all = torch.cat(x_hat_all, dim=1)
        reconstruction_loss += z_gap_loss * 1e-5  


        ##################################
        # Step 3: Sparsity loss 
        ##################################
        mask = torch.tril(torch.ones(self.num_latent, self.num_latent), diagonal=-1).to(args.device)
        matrix = self.A * mask
        # print('matrix: ', matrix.shape)
        # print('mask: ', mask.shape)
        # print('self.num_latent: ', self.num_latent)
        # print('A: ', self.A.shape)
        A_blocks = []
        for i in range(1, self.n_modalities):
            for j in range(i):
                row_start, row_end = i * self.l_dim, (i + 1) * self.l_dim
                col_start, col_end = j * self.l_dim, (j + 1) * self.l_dim
                A_blocks.append(matrix[row_start:row_end, col_start:col_end])
        A_xy = torch.cat(A_blocks, dim=0)
        A_xy_est = A_xy * (A_xy>self.alpha_threshold).float()
        num_connections = torch.sum(A_xy_est != 0).float()
        if num_connections < self.l_dim:
            connection_penalty = (num_connections - self.l_dim*(self.n_modalities-1)) ** 2
        else:
            connection_penalty = 0.0
        sparsity_loss = torch.norm(A_xy, p=1) + connection_penalty
        # print('A xy est: ', A_xy_est.shape)

        ###################################################
        # Step 4: KL loss for Z to match with epsilon
        ###################################################
        # 1. Only consider the estimated parents of Z
        # mid = self.transfer(args, 【z_all， s】, matrix)
        mid = self.transfer(args, z_all, matrix)
        # print('mid: ', mid.shape)
        eps_all, log_det = self.updatee(mid) # 
        kld_loss = flow_kl_divergence(args, eps_all, eta_all, log_det)
        
        # # 2. Consider all influence from Z
        # eps = z_all - torch.matmul(z_all, matrix.T) - self.b
        # for i in range(self.num_latent):
        #     eps_i = eps[:, i].unsqueeze(1)
        #     eps_i = self.updatee[i](eps_i)
        #     eps_group.append(eps_i)
        # eps_all = torch.cat(eps_group, dim=1)
        # kld_loss = calculate_kl_divergence(eps_all, eta_all)

        return z_all, None, reconstruction_loss, kld_loss, sparsity_loss
