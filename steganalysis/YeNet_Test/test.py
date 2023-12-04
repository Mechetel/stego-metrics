"""This module is used to test the YeNet model."""
from glob import glob
import torch
import numpy as np
import imageio.v2 as io
from model.model import YeNet

COVER_PATH = "../../images/*"
STEGO_PATH = "../../encoded_images/Tiger/*"
CHKPT = "./checkpoints/net_357.pt"

def load_model():
    model = YeNet().cpu()
    ckpt = torch.load(CHKPT, map_location='cpu')
    model.load_state_dict(ckpt["model_state_dict"])
    return model

def load_image(image_path):
    image = torch.empty(1, 1, 256, 256, dtype=torch.float)
    image[0, 0, :, :] = torch.tensor(io.imread(image_path)).cpu()
    return image.cpu()

def test_accuracy(model, cover_paths, stego_paths):
    test_accuracy = []

    for cover_path, stego_path in zip(cover_paths, stego_paths):
        cover_image = load_image(cover_path)
        stego_image = load_image(stego_path)

        cover_label = torch.tensor([0], dtype=torch.long).cpu()
        stego_label = torch.tensor([1], dtype=torch.long).cpu()

        cover_output = model(cover_image)
        stego_output = model(stego_image)

        cover_prediction = cover_output.data.max(1)[1]
        stego_prediction = stego_output.data.max(1)[1]

        cover_accuracy = cover_prediction.eq(cover_label.data).sum().item() * 100.0 / cover_label.size()[0]
        stego_accuracy = stego_prediction.eq(stego_label.data).sum().item() * 100.0 / stego_label.size()[0]

        test_accuracy.append((cover_accuracy, stego_accuracy))

    return test_accuracy

def main():
    cover_image_names = glob(COVER_PATH)
    stego_image_names = glob(STEGO_PATH)

    cover_labels = np.zeros((len(cover_image_names)))
    stego_labels = np.ones((len(stego_image_names)))

    model = load_model()
    test_accuracy_values = test_accuracy(model, cover_image_names, stego_image_names)

    for idx, (cover_accuracy, stego_accuracy) in enumerate(test_accuracy_values, start=1):
        print(f"Image {idx}: Cover Accuracy: {cover_accuracy:.2f}, Stego Accuracy: {stego_accuracy:.2f}")

if __name__ == "__main__":
    main()
