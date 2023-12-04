import sys
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np

from packages.processor import PVDProcessor
from packages.metrics import (
        calculate_metrics,
        calculate_symbols,
        calculate_pixels,
        generate_random_string
        )


# imtxt = "Lenna"
imtxt = "Tiger"
# imtxt = "Garage"
# imtxt = "Owls"
# imtxt = "Desert"
# imtxt = "City"
bpp_values = [0.01, 0.1, 1, 1.5, 2, 2.5, 3, 3.5, 4]

for bpp_value in bpp_values:
    filename = './images/{0}.png'.format(imtxt)
    filename_encoded = './encoded_images/{0}/{1}_pvd_encoded.png'.format(imtxt, bpp_value)

    symbols_count = calculate_symbols(bpp_value, calculate_pixels(cv2.imread(filename)))
    message = generate_random_string(symbols_count)
    print("Message length:", len(message), "symbols")


    text_filename = './texts/secret_text_file.txt'
    text_filename_out = './texts/secret_text_file_out.txt'
    with open(text_filename, 'w') as f:
        f.write(message)

    pvd_embed = PVDProcessor(filename)
    pvd_embed.embed_payload(text_filename, filename_encoded)
    # pvd_extract = PVDProcessor(filename_encoded)
    # extracted_message = pvd_extract.extract_payload(text_filename_out, xor_key = 0)
    # f = open(text_filename_out, "r")
    # print("Messages is equal?", message == f.read())


    image1 = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(cv2.imread(filename_encoded), cv2.COLOR_BGR2RGB)

    bpp, ssim, mse, psnr = calculate_metrics(image1, image2, message)
    print("BPP:  ", format(float(bpp),  '.10f'))
    print("SSIM: ", format(float(ssim), '.10f'))
    print("MSE:  ", format(float(mse),  '.10f'))
    print("PSNR: ", format(float(psnr), '.10f'))
