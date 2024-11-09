import torch
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

def train_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
               data: torch.utils.data.DataLoader, loss_fn, device: torch.device):
    model.train()
    train_loss = 0
    all_preds = []
    all_targets = []

    for batch_idx, (features, target) in enumerate(data):
        features, target = features.to(device), target.to(device)

        # Forward pass and ensure y_pred shape matches target
        y_pred = model(features).squeeze(-1)  # Removing extra dimension to match target shape

        # Compute the loss (MSE)
        loss = loss_fn(y_pred, target)
        train_loss += loss.item()

        # Accumulate predictions and targets for calculating MAE and RMSE
        all_preds.append(y_pred.detach().cpu())
        all_targets.append(target.cpu())

        # Zero the gradients, perform backward pass, and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Concatenate predictions and targets for metric calculation
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Calculate metrics
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = root_mean_squared_error(all_targets, all_preds)  # Using the new RMSE function

    train_loss = train_loss / len(data)  # Average MSE loss
    return train_loss, mse, mae, rmse

def eval_step(model: torch.nn.Module, data: torch.utils.data.DataLoader, loss_fn, device: torch.device):
    model.eval()
    total_test_loss = 0
    all_preds = []
    all_targets = []

    with torch.inference_mode():
        for batch_idx, (features, target) in enumerate(data):
            features, target = features.to(device), target.to(device)

            # Forward pass and ensure y_pred shape matches target
            y_pred = model(features).squeeze(-1)  # Removing extra dimension to match target shape

            # Compute the loss (MSE)
            loss = loss_fn(y_pred, target)
            total_test_loss += loss.item()

            # Accumulate predictions and targets for calculating MAE and RMSE
            all_preds.append(y_pred.cpu())
            all_targets.append(target.cpu())

    # Concatenate predictions and targets for metric calculation
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Calculate metrics
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = root_mean_squared_error(all_targets, all_preds)  # Using the new RMSE function

    avg_test_loss = total_test_loss / len(data)  # Average MSE loss
    return avg_test_loss, mse, mae, rmse

def train(model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          loss_fn: torch.nn.MSELoss,  # Regression loss function
          train_loader: torch.utils.data.DataLoader,
          test_loader: torch.utils.data.DataLoader,
          patience: int = 10):  # patience parameter to enable early stopping

    result = {
        "train_loss": [],
        "train_mse": [],
        "train_mae": [],
        "train_rmse": [],
        "test_loss": [],
        "test_mse": [],
        "test_mae": [],
        "test_rmse": []
    }

    best_test_loss = float('inf')  # Initialize with a large value
    epochs_without_improvement = 0  # Counter for early stopping

    for epoch in tqdm(range(epochs)):
        # Training step
        train_loss, train_mse, train_mae, train_rmse = train_step(
            model=model,
            optimizer=optimizer,
            data=train_loader,
            device=device,
            loss_fn=loss_fn
        )

        # Evaluation step
        test_loss, test_mse, test_mae, test_rmse = eval_step(
            model=model,
            data=test_loader,
            loss_fn=loss_fn,
            device=device
        )

        # Print results for this epoch
        print(f"\nEpoch {epoch + 1} | "
              f"Train MSE: {train_mse:.4f}, Train MAE: {train_mae:.4f}, Train RMSE: {train_rmse:.4f} | "
              f"Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}")

        # Record results
        result["train_loss"].append(train_loss)
        result["train_mse"].append(train_mse)
        result["train_mae"].append(train_mae)
        result["train_rmse"].append(train_rmse)
        result["test_loss"].append(test_loss)
        result["test_mse"].append(test_mse)
        result["test_mae"].append(test_mae)
        result["test_rmse"].append(test_rmse)

        # Early Stopping Check
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_without_improvement = 0  # Reset the counter if there's improvement
        else:
            epochs_without_improvement += 1

        # Stop training if no improvement for 'patience' epochs
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    return result

import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

def evaluate_model(model: torch.nn.Module,
                  test_loader: torch.utils.data.DataLoader,
                  loss_fn: torch.nn.Module,
                  device: torch.device):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_targets = []
    total_test_loss = 0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for features, target in test_loader:
            features, target = features.to(device), target.to(device)

            # Forward pass to get predictions
            y_pred = model(features).squeeze(-1)  # Remove extra dimension to match target shape

            # Compute the loss (MSE)
            loss = loss_fn(y_pred, target)
            total_test_loss += loss.item()

            # Accumulate predictions and targets for metric calculation
            all_preds.append(y_pred.cpu())
            all_targets.append(target.cpu())

    # Concatenate all predictions and targets for metric calculation
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Calculate metrics
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = mean_squared_error(all_targets, all_preds, squared=False)  # RMSE directly from MSE

    # Compute average test loss in the same way as global MSE
    avg_test_loss = mean_squared_error(all_targets, all_preds)

    # Print evaluation metrics
    print(f"\nEvaluation Results: "
          f"Test Loss (MSE): {avg_test_loss:.4f} | "
          f"Test MSE: {mse:.4f} | "
          f"Test MAE: {mae:.4f} | "
          f"Test RMSE: {rmse:.4f}")

    return {
        "test_loss": avg_test_loss,
        "test_mse": mse,
        "test_mae": mae,
        "test_rmse": rmse
    }

