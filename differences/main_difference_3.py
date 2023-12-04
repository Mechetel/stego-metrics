import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 10

def adjust_brightness_contrast(image, alpha, beta):
    return cv2.addWeighted(image, alpha, image, 0, beta)


# imtxt = "Lenna"
imtxt = "Tiger"
# imtxt = "Garage"
# imtxt = "Owls"
# imtxt = "Desert"
# imtxt = "City"
bpp_values = [1, 1.5, 2, 2.5, 3, 3.5, 4]

for bpp_value in bpp_values:
    filename_original           = './images/{0}.png'.format(imtxt)
    filename_encoded_lsb        = './encoded_images/{0}/{1}_lsb_encoded.png'.format(imtxt, bpp_value)
    filename_encoded_steganogan = './encoded_images/{0}/{1}_steganogan_encoded.png'.format(imtxt, bpp_value)
    filename_encoded_pvd        = './encoded_images/{0}/{1}_pvd_encoded.png'.format(imtxt, bpp_value)

    # load images
    image_original           = cv2.imread(filename_original, 1)
    image_encoded_lsb        = cv2.imread(filename_encoded_lsb, 1)
    image_encoded_steganogan = cv2.imread(filename_encoded_steganogan, 1)
    image_encoded_pvd        = cv2.imread(filename_encoded_pvd, 1)

    # resize to shape of original
    image_original           = cv2.resize(image_original, (image_original.shape[1], image_original.shape[0]))
    image_encoded_lsb        = cv2.resize(image_encoded_lsb, (image_original.shape[1], image_original.shape[0]))
    image_encoded_steganogan = cv2.resize(image_encoded_steganogan, (image_original.shape[1], image_original.shape[0]))
    image_encoded_pvd        = cv2.resize(image_encoded_pvd, (image_original.shape[1], image_original.shape[0]))


    # to rgb
    rgb_image_original           = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    rgb_image_encoded_lsb        = cv2.cvtColor(image_encoded_lsb, cv2.COLOR_BGR2RGB)
    rgb_image_encoded_steganogan = cv2.cvtColor(image_encoded_steganogan, cv2.COLOR_BGR2RGB)
    rgb_image_encoded_pvd        = cv2.cvtColor(image_encoded_pvd, cv2.COLOR_BGR2RGB)

    # to grayscale
    gray_image_original           = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
    gray_image_encoded_lsb        = cv2.cvtColor(image_encoded_lsb, cv2.COLOR_BGR2GRAY)
    gray_image_encoded_steganogan = cv2.cvtColor(image_encoded_steganogan, cv2.COLOR_BGR2GRAY)
    gray_image_encoded_pvd        = cv2.cvtColor(image_encoded_pvd, cv2.COLOR_BGR2GRAY)


    # compute difference
    gray_difference_original           = cv2.subtract(gray_image_original, gray_image_original)
    gray_difference_encoded_lsb        = cv2.subtract(gray_image_original, gray_image_encoded_lsb)
    gray_difference_encoded_steganogan = cv2.subtract(gray_image_original, gray_image_encoded_steganogan)
    gray_difference_encoded_pvd        = cv2.subtract(gray_image_original, gray_image_encoded_pvd)

    rgb_difference_original           = cv2.subtract(rgb_image_original, rgb_image_original)
    rgb_difference_encoded_lsb        = cv2.subtract(rgb_image_original, rgb_image_encoded_lsb)
    rgb_difference_encoded_steganogan = cv2.subtract(rgb_image_original, rgb_image_encoded_steganogan)
    rgb_difference_encoded_pvd        = cv2.subtract(rgb_image_original, rgb_image_encoded_pvd)


    _, binarized_gray_difference_original           = cv2.threshold(gray_difference_original,           1, 255, cv2.THRESH_BINARY)
    _, binarized_gray_difference_encoded_lsb        = cv2.threshold(gray_difference_encoded_lsb,        1, 255, cv2.THRESH_BINARY)
    _, binarized_gray_difference_encoded_steganogan = cv2.threshold(gray_difference_encoded_steganogan, 1, 255, cv2.THRESH_BINARY)
    _, binarized_gray_difference_encoded_pvd        = cv2.threshold(gray_difference_encoded_pvd,        1, 255, cv2.THRESH_BINARY)

    _, binarized_rgb_difference_original            = cv2.threshold(rgb_difference_original,            1, 255, cv2.THRESH_BINARY)
    _, binarized_rgb_difference_encoded_lsb         = cv2.threshold(rgb_difference_encoded_lsb,         1, 255, cv2.THRESH_BINARY)
    _, binarized_rgb_difference_encoded_steganogan  = cv2.threshold(rgb_difference_encoded_steganogan,  1, 255, cv2.THRESH_BINARY)
    _, binarized_rgb_difference_encoded_pvd         = cv2.threshold(rgb_difference_encoded_pvd,         1, 255, cv2.THRESH_BINARY)


    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    for ax in axes.ravel():
        ax.axis('off')

    axes[0, 0].imshow(rgb_image_original)
    axes[0, 0].set_title("Original Image")
    axes[0, 1].imshow(rgb_image_encoded_lsb)
    axes[0, 1].set_title("LSB encoded Image")
    axes[0, 2].imshow(rgb_image_encoded_steganogan)
    axes[0, 2].set_title("STEGANOGAN encoded Image")
    axes[0, 3].imshow(rgb_image_encoded_pvd)
    axes[0, 3].set_title("PVD encoded Image")

    axes[1, 0].imshow(binarized_rgb_difference_original)
    axes[1, 0].set_title("Binarized Diff Original Image")
    axes[1, 1].imshow(binarized_rgb_difference_encoded_lsb)
    axes[1, 1].set_title("Binarized Diff LSB encoded Image")
    axes[1, 2].imshow(binarized_rgb_difference_encoded_steganogan)
    axes[1, 2].set_title("Binarized Diff STEGANOGAN encoded Image")
    axes[1, 3].imshow(binarized_rgb_difference_encoded_pvd)
    axes[1, 3].set_title("Binarized Diff PVD encoded Image")

    axes[2, 0].imshow(binarized_gray_difference_original, cmap='gray')
    axes[2, 0].set_title("Binarized Diff Gray Original Image")
    axes[2, 1].imshow(binarized_gray_difference_encoded_lsb, cmap='gray')
    axes[2, 1].set_title("Binarized Diff Gray LSB encoded Image")
    axes[2, 2].imshow(binarized_gray_difference_encoded_steganogan, cmap='gray')
    axes[2, 2].set_title("Binarized Diff Gray STEGANOGAN encoded Image")
    axes[2, 3].imshow(binarized_gray_difference_encoded_pvd, cmap='gray')
    axes[2, 3].set_title("Binarized Diff Gray PVD encoded Image")

    plt.subplots_adjust(left=0.004, bottom=0, right=0.996, top=1, wspace=0.02, hspace=0)
    filename_to_save = "./differences/main/{0}/{1}_diff.png".format(imtxt, bpp_value)
    plt.savefig(filename_to_save)
