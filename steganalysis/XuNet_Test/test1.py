from glob import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model.model import XuNet

COVER_PATH = "../../images/Tiger/Tiger.png"
STEGO_PATH = "../../encoded_images/Tiger/*"
CHKPT = "./checkpoints/net_437.pt"

resize = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
])

def load_model(ckpt_path):
    model = XuNet()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt["model_state_dict"])
    return model

def main():
    cover_image = Image.open(COVER_PATH)
    cover_image = resize(cover_image)

    stego_image_names = glob(STEGO_PATH)

    model = load_model(CHKPT)

    images = torch.empty((len(stego_image_names), 1, 256, 256), dtype=torch.float)
    test_accuracy = []

    for i, stego_path in enumerate(stego_image_names):
        stego_image = Image.open(stego_path)
        stego_image = resize(stego_image)
        images[i, 0, :, :] = stego_image.squeeze(0)

    cover_tensor = cover_image.unsqueeze(0)
    image_tensor = images

    outputs = model(image_tensor, torch.device('cpu'))
    predictions = outputs.data.max(1)[1]

    for stego_path, prediction in zip(stego_image_names, predictions):
        print(f"Stego Image: {stego_path}, Predicted Class: {prediction.item()}")

    accuracy = predictions.eq(0).sum().item() * 100.0 / len(predictions)
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
