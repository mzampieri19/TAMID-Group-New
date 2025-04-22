import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

class CSVObjectDetectionDataset(Dataset):
    def __init__(self, images_dir, csv_path, class_map):
        self.images_dir = images_dir
        self.df = pd.read_csv(csv_path)
        self.class_map = class_map
        self.image_files = self.df['filename'].unique()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = image.resize((512, 512))
        image_tensor = F.to_tensor(image)

        boxes_df = self.df[self.df['filename'] == img_name]
        boxes = []
        labels = []

        for _, row in boxes_df.iterrows():
            x_scale = 512 / row['width']
            y_scale = 512 / row['height']
            xmin = row['xmin'] * x_scale
            ymin = row['ymin'] * y_scale
            xmax = row['xmax'] * x_scale
            ymax = row['ymax'] * y_scale
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_map[row['class']])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        return image_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    class_map = {
        'botellas': 1, 
        'HDPE': 2, 
        'Latas': 3, 
        'LDPE': 4, 
        'null': 5,
        'OTHERS': 6,
        'paper': 7,
        'PET': 8,
        'Plastic': 9,
        'PP': 10,
        'PS': 11,
        }    
    num_classes = len(class_map) + 1

    dataset = CSVObjectDetectionDataset(
        images_dir="/Users/michelangelozampieri/Desktop/TAMID-Group-New/data/Waste segregation.v1i.tensorflow/train",
        csv_path="/Users/michelangelozampieri/Desktop/TAMID-Group-New/data/Waste segregation.v1i.tensorflow/train/_annotations.csv",
        class_map=class_map
    )

    loader = DataLoader(
        dataset, batch_size=4, shuffle=True,
        num_workers=4, pin_memory=True,
        collate_fn=collate_fn
    )

    model = ssd300_vgg16(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    scaler = GradScaler()

    for epoch in range(10):
        model.train()
        total_loss = 0

        for images, targets in tqdm(loader, desc=f"Epoch {epoch+1}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with autocast():
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

if __name__ == '__main__':
    main()
