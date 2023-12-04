import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 10

def adjust_brightness_contrast(image, alpha, beta):
    return cv2.addWeighted(image, alpha, image, 0, beta)

bpp_values = [0.01, 0.1, 1, 1.5, 2, 2.5, 3, 3.5, 4]

# Define subplots for original images, binarized_rgb_difference_encoded_lsb, and binarized_gray_difference_encoded_lsb
fig, axes = plt.subplots(3, len(bpp_values), figsize=(14, 5))
for ax in axes.ravel():
    ax.axis('off')

# Iterate over bpp values
for i, bpp_value in enumerate(bpp_values):
    imtxt = "Tiger"  # You can change the image name as needed

    # Load images
    filename_original = './images/{0}.png'.format(imtxt)
    filename_encoded_steganogan = './encoded_images/{0}/{1}_steganogan_encoded.png'.format(imtxt, bpp_value)

    # load images
    image_original    = cv2.imread(filename_original, 1)
    image_encoded_steganogan = cv2.imread(filename_encoded_steganogan, 1)

    # resize to shape of original
    image_original    = cv2.resize(image_original, (image_original.shape[1], image_original.shape[0]))
    image_encoded_steganogan = cv2.resize(image_encoded_steganogan, (image_original.shape[1], image_original.shape[0]))

    # to rgb
    rgb_image_original     = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    rgb_image_encoded_steganogan  = cv2.cvtColor(image_encoded_steganogan, cv2.COLOR_BGR2RGB)

    # to grayscale
    gray_image_original    = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
    gray_image_encoded_steganogan = cv2.cvtColor(image_encoded_steganogan, cv2.COLOR_BGR2GRAY)

    # Compute differences
    gray_difference_encoded_steganogan = cv2.subtract(gray_image_original, gray_image_encoded_steganogan)
    rgb_difference_encoded_steganogan  = cv2.subtract(rgb_image_original, rgb_image_encoded_steganogan)

    _, binarized_gray_difference_encoded_steganogan = cv2.threshold(gray_difference_encoded_steganogan, 1, 255, cv2.THRESH_BINARY)
    _, binarized_rgb_difference_encoded_steganogan = cv2.threshold(rgb_difference_encoded_steganogan, 1, 255, cv2.THRESH_BINARY)

    # Plot images and add subtitles to each row
    axes[0, i].set_title("BPP: {}".format(bpp_value))
    axes[0, i].imshow(rgb_image_encoded_steganogan)

    axes[1, 4].set_title("Binarized RGB Difference")
    axes[1, i].imshow(binarized_rgb_difference_encoded_steganogan, cmap='gray')

    axes[2, 4].set_title("Binarized Gray Difference")
    axes[2, i].imshow(binarized_gray_difference_encoded_steganogan, cmap='gray')

# Adjust layout and save the figure
plt.subplots_adjust(left=0.004, bottom=0, right=0.996, top=0.9, wspace=0.02, hspace=0.2)
filename_to_save = "./differences/steganogan/{0}_diff.png".format(imtxt)
plt.savefig(filename_to_save)
# plt.show()
