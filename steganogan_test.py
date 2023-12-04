import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np

from steganogan import SteganoGAN
from packages.metrics import (
        calculate_symbols,
        calculate_pixels,
        generate_random_string
        )

from skimage.metrics import (
        structural_similarity,
        peak_signal_noise_ratio,
        mean_squared_error
        )

def calculate_metrics(original_image, encoded_image, message=''):
    bpp = len(message) * 8 / (original_image.shape[0] * original_image.shape[1])
    ssim, _ = structural_similarity(original_image, encoded_image, full=True, multichannel=True)
    mse = mean_squared_error(original_image, encoded_image)
    psnr = peak_signal_noise_ratio(original_image, encoded_image)
    return bpp, ssim, mse, psnr


# imtxt = "Lenna"
imtxt = "Tiger"
# imtxt = "Garage"
# imtxt = "Owls"
# imtxt = "Desert"
# imtxt = "City"

"""WARNING MAY PASS FOR VERY LONG TIME"""
"""BETTER TO RUN EACH BPP IN PARALLEL"""
bpp_values = [0.01, 0.1, 1, 1.5, 2, 2.5, 3, 3.5, 4]

for bpp_value in bpp_values:
    filename = './images/{0}.png'.format(imtxt)
    filename_encoded = './encoded_images/{0}/{1}_steganogan_encoded.png'.format(imtxt, bpp_value)

    symbols_count = calculate_symbols(bpp_value, calculate_pixels(cv2.imread(filename)))
    message = generate_random_string(symbols_count)
    print("Message length:", len(message), "symbols")

    steganogan = SteganoGAN.load(architecture='basic')
    steganogan.encode(filename, filename_encoded, message)
    # print("Messages is equal?", message == steganogan.decode(filename_encoded))


    image1 = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(cv2.imread(filename_encoded), cv2.COLOR_BGR2RGB)

    bpp, ssim, mse, psnr = calculate_metrics(image1, image2, message)
    print("BPP:  ", format(float(bpp),  '.10f'))
    print("SSIM: ", format(float(ssim), '.10f'))
    print("MSE:  ", format(float(mse),  '.10f'))
    print("PSNR: ", format(float(psnr), '.10f'))
