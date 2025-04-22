import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import create_model
from train import train_model
from test import test_model
from predict import predict_image
from visualize import plot_confusion_matrix

if __name__ == "__main__":
    # --- Setup ---
    # Define paths
    train_dir = "/Users/michelangelozampieri/Desktop/TAMID-Group-New/data/sorted_data_output/train"
    test_dir = "/Users/michelangelozampieri/Desktop/TAMID-Group-New/data/sorted_data_output/test"
    model_dir = "/Users/michelangelozampieri/Desktop/TAMID-Group-New/models"

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Preparation ---
    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Just normalization for validation
    transform_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])  

    # Dataset and DataLoader
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_eval)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # --- Model Training and Evaluation ---
    # Model, loss, optimizer, and scheduler
    class_names = train_dataset.classes
    num_classes = len(class_names)
    model = create_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Train the model
    num_epochs = 10
    model = train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs)

    # Test the model
    test_model(model, test_loader, criterion, device)

    # --- Visualization ---
    # Plot confusion matrix
    plot_confusion_matrix(model, test_loader, class_names, device)

    # --- Save and Predict ---
    # Save the model
    model_path = os.path.join(model_dir, "improved_resnet18.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Predict an image
    image_path = "path_to_image.jpg"
    predicted_class = predict_image(image_path, model, class_names, device)
    print(f"Predicted class: {predicted_class}")