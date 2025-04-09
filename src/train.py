
import torch
from tqdm import tqdm

def train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs):
    """
    Trains the model.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        device (torch.device): Device to train on (CPU or GPU).
        num_epochs (int): Number of epochs.

    Returns:
        nn.Module: The trained model.
    """
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        with tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix(loss=running_loss / (total / labels.size(0)), accuracy=100. * correct / total)

        scheduler.step()
        print(f"Epoch {epoch+1}: Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100.*correct/total:.2f}%")

    return model