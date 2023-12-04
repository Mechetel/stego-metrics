import sys
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np

from packages.DCT import DCT
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
bpp_values = [0.01, 0.1]

for bpp_value in bpp_values:
    filename = './images/{0}.png'.format(imtxt)
    filename_encoded = './encoded_images/{0}/{1}_dct_encoded.png'.format(imtxt, bpp_value)

    symbols_count = calculate_symbols(bpp_value, calculate_pixels(cv2.imread(filename)))
    message = generate_random_string(symbols_count)
    print("Message length:", len(message), "symbols")

    x = DCT(filename)
    secret = x.DCTEn(message, filename_encoded)
    # y = DCT(filename_encoded)
    # print("Messages is equal?", message == y.DCTDe())


    image1 = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(cv2.imread(filename_encoded), cv2.COLOR_BGR2RGB)
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    bpp, ssim, mse, psnr = calculate_metrics(image1, image2, message)
    print("BPP:  ", format(float(bpp),  '.10f'))
    print("SSIM: ", format(float(ssim), '.10f'))
    print("MSE:  ", format(float(mse),  '.10f'))
    print("PSNR: ", format(float(psnr), '.10f'))
