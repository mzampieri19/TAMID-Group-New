import torch
from torchvision import transforms
from PIL import Image

def predict_image(image_path, model, class_names, device):
    """
    Predicts the class of an image using the trained model.

    Args:
        image_path (str): Path to the image file.
        model (torch.nn.Module): Trained PyTorch model.
        class_names (list): List of class names corresponding to the model's output.
        device (torch.device): Device to run the model on (CPU or GPU).

    Returns:
        str: Predicted class name.
    """
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = outputs.max(1)

    return class_names[predicted.item()]