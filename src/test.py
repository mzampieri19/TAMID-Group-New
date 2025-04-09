import torch
from tqdm import tqdm

def test_model(model, test_loader, criterion, device):
    """
    Tests the model.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for testing data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to test on (CPU or GPU).

    Returns:
        tuple: Test loss and accuracy.
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with tqdm(test_loader, desc="Testing", unit="batch") as pbar:
        with torch.no_grad():
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix(loss=test_loss / (total / labels.size(0)), accuracy=100. * correct / total)

    test_loss = test_loss / len(test_loader)
    test_accuracy = 100. * correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    return test_loss, test_accuracy