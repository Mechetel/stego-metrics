import os, sys
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np

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
    filename_encoded = './encoded_images/{0}/{1}_lsb_encoded.png'.format(imtxt, bpp_value)

    symbols_count = calculate_symbols(bpp_value, calculate_pixels(cv2.imread(filename)))
    message = generate_random_string(symbols_count)
    print("Message length:", len(message), "symbols")


    text_filename = './texts/secret_text_file.txt'
    text_filename_out = './texts/secret_text_file_out.txt'
    with open(text_filename, 'w') as f:
        f.write(message)

    os_command_to_encode = "stegolsb steglsb -h -i {0} -s {1} -o {2} -n 2".format(filename, text_filename, filename_encoded)
    # os_command_to_decode = "stegolsb steglsb -r -i {0} -o {1} -n 1".format(filename_encoded, text_filename_out)
    os.system(os_command_to_encode)
    # os.system(os_command_to_decode)

    # f = open(text_filename_out, "r")
    # print("Messages is equal?", message == f.read())


    # OTHER METHOD
    # from stegano import lsb
    # secret = lsb.hide(filename, message)
    # secret.save(filename_encoded)
    # print("Messages is equal?", message == lsb.reveal(filename_encoded))


    # OTHER METHOD
    # from packages.processor import LSBProcessor
    # text_filename = './texts/secret_text_file.txt'
    # text_filename_out = './texts/secret_text_file_out.txt'
    # with open(text_filename, 'w') as f:
    #     f.write(message)
    # lsb_embed = LSBProcessor(filename)
    # lsb_embed.embed_payload(text_filename, filename_encoded)
    # lsb_extract = LSBProcessor(filename_encoded)
    # lsb_extract.extract_payload(text_filename_out)
    # f = open(text_filename_out, "r")
    # print("Messages is equal?", message == f.read())


    # OTHER METHOD
    # from packages.LSB import LSB
    # x = LSB(filename)
    # encoded = x.hide(message, filename_encoded)
    # y = LSB(filename_encoded)
    # print("Messages is equal?", message == y.extract())


    image1 = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(cv2.imread(filename_encoded), cv2.COLOR_BGR2RGB)

    bpp, ssim, mse, psnr = calculate_metrics(image1, image2, message)
    print("BPP:  ", format(float(bpp),  '.10f'))
    print("SSIM: ", format(float(ssim), '.10f'))
    print("MSE:  ", format(float(mse),  '.10f'))
    print("PSNR: ", format(float(psnr), '.10f'))
