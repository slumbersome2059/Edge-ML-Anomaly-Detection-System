import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
#nn contains layers(transform input into something useful later), models(how to give data between layers),...
class ConvAutoencoder1D(nn.Module):
    """Ultra-lightweight 1D Conv Autoencoder (<1,200 parameters).

    Input Shape:  (Batch, 4, 20)
    Bottleneck:   (Batch, 8, 20)
    Output Shape: (Batch, 4, 20)
    """

    def __init__(self, in_channels: int = 5):
        super().__init__()
        """
        Encoder

        - For each batch, the conv1D take the 20x5 matrix(window_size x feature_size) matrix and 
        it applies 16 different filters(just matrices that contained weights that can be trained) of weights of 3x5.
        - A filter of 3x1 applied to 20x1(call it a) will produce [filt dot a[0:3], filt dot a[1:4], ...]
        - I think if you have 3xn(f) applied to 20xn(a) this is f.col(0) applied to 
        a.col(0) + f.col(1) applied to a.col(1) + ...(the addition is columnwise)  
        - One extra 0 value is added to either side of the window(this is the padding) 
        - Because without padding your window size decreases because the border information
        (info at the very left and right of window ) is lost and having padding allows you to 
        come up with some kind of approximation to the lost information which maintains window size
        - Kernel size is the number of rows in the filters
        - We are expanding to 16 features so that we give the opportunity for the NN 
        to learn relationships between our 5 features(link between RPM and SPEED) and also relationships across 
        kernel size for one feature(acceleration rate) 
        - The sequential takes input and gives it to layer and its output is directly 
        fed into next layer and so on till it gives output
        """
        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels, 16, kernel_size=3, padding=1
            ),  #(B, 16, 20)
            nn.ReLU(),#this makes negative numbers 0, and leaves positive ones the same(this non-linearity allows it to model complex relationships between features)
            nn.Conv1d(16, 8, kernel_size=3, padding=1),  # Bottleneck: (B, 8, 20)
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3, padding=1),  # (B, 16, 20)
            nn.ReLU(),
            nn.Conv1d(16, in_channels, kernel_size=3, padding=1),  # (B, 4, 20)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

def compute_reconstruction_errors(
    model: nn.Module, data_tensor: torch.Tensor
) -> np.ndarray:
    """
    Computes MSE reconstruction loss per window sample.
    
    DataTensor is a list of matrices(each matrices rows of features and columns of window index)
    """
    model.eval()#setting to evaluation mode, basically saying no training happens and just evaluation
    with torch.no_grad():#disables gradient calculation
        inputs = data_tensor.to(device)#this is for transferring to GPU but I'm not using a GPU so this code is redundant and I think data_tensor will be assigned to inputs
        reconstructions = model(inputs)
        # Mean squared error aggregated over features and window dimensions
        mse_per_sample = (
            torch.mean((inputs - reconstructions) ** 2, dim=(1, 2))
            .cpu()#sends it to CPU(does nothing if already in CPU)
            .numpy()
        )
        # We are subtracting two lists of matrices and then squaring each entry in matrix 
        # and then taking a mean of all values in one matrix to get one value
        # The final output is an np array
        # The dim=(1,2) specifies the dimensions to reduce so 1 is rows and 2 is columns
        # When reducing rows dimension you would do sum(column)/column_size, same thing applies for reducing columns, 
        # this is same as taking mean of all values in matrix in one go no matter how you do it
    return mse_per_sample


def train_autoencoder(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 25
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #something that will optimise parameters(things like weights and biases) on every batch size
    #hyperparameters are different to parameters, hyperparameters control the 
    # optimisation of parameters -> hps are the num of epochs, 
    # learning rate(affect learning speed) given by lr 

    model.to(device)

    max_epochs = epochs
    patience = 3
    min_delta = 0.01

    best_val_loss = float("inf")
    best_model_weights = None
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()#set to training mode
        train_loss = 0.0
        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()#the optimiser still has the gradients calculated by loss.backward()
            #without loss.backward might add to the old gradients that were stored so step would overdo its adjustments
            outputs = model(x_batch)
            loss = criterion(outputs, x_batch)#outputs should be reconstruction of x_batch
            loss.backward()#does backpropogation and calculate gradients with respect to parameters and deposits them to be used by optimiser
            optimizer.step()#updates parameters by using gradients collected
            train_loss += loss.item() * x_batch.size(0) #loss is an one element tensor and item() gets its value(its an average of the MSE from each input in batch)
            # you want to get the total sum of loss because we divide by the length of the whole dataset
            # doing it this way is more accurate when batch_size does not go into dataset size

        train_loss /= len(train_loader.dataset)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x_val,) in val_loader:
                x_val = x_val.to(device)
                preds = model(x_val)
                loss = criterion(preds, x_val)
                val_loss += loss.item() * x_val.size(0)
        val_loss /= len(val_loader.dataset)

        print(
                f"Epoch {epoch:02d}/{max_epochs:02d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}"
            )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    # Restore best performing model weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

def calculate_anomaly_threshold(
    model: nn.Module, val_tensor: torch.Tensor, n_sigmas: float = 3.0
):
    """Calculates statistical reconstruction error thresholds on clean validation data."""
    val_errors = compute_reconstruction_errors(model, val_tensor)

    mean_err = np.mean(val_errors)
    std_err = np.std(val_errors)

    # 3-Sigma Rule Threshold
    sigma_threshold = mean_err + (n_sigmas * std_err)
    # 99th Percentile Threshold (Alternative non-parametric limit)
    p99_threshold = np.percentile(val_errors, 99)

    print("\n--- Validation Reconstruction Error Baseline ---")
    print(f"Mean Error: {mean_err:.6f}")
    print(f"Std Dev:    {std_err:.6f}")
    print(f"3-Sigma Threshold:    {sigma_threshold:.6f}")
    print(f"99th Pct Threshold:   {p99_threshold:.6f}")

    return sigma_threshold, p99_threshold