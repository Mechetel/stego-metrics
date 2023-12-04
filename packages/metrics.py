import random
import string
import numpy as np
from skimage.metrics import (
        structural_similarity,
        peak_signal_noise_ratio,
        mean_squared_error
        )

def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

def calculate_metrics(original_image, encoded_image, message=''):
    bpp = len(message) * 8 / (original_image.shape[0] * original_image.shape[1])
    ssim, _ = structural_similarity(original_image, encoded_image, full=True, channel_axis=-1)
    mse = mean_squared_error(original_image, encoded_image)
    psnr = peak_signal_noise_ratio(original_image, encoded_image)
    return bpp, ssim, mse, psnr

def calculate_symbols(bpp, total_pixels):
    bits_per_pixel = bpp * total_pixels
    bytes_needed = round(bits_per_pixel / 8)
    return bytes_needed

def calculate_pixels(image):
    height, width, _ = image.shape
    return height * width
