import cv2
import numpy as np
from skimage.metrics import structural_similarity

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 8



# imtxt = "Lenna"
imtxt = "Tiger"
# imtxt = "Garage"
# imtxt = "Owls"
# imtxt = "Desert"
# imtxt = "City"

y1_dct_values = np.array([])
y2_lsb_values = np.array([])
y3_steganogan_values = np.array([])
y4_pvd_values = np.array([])

x_bpp_values = [0.01, 0.1, 1, 1.5, 2, 2.5, 3, 3.5, 4]

for bpp_value in x_bpp_values:
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

    y2_lsb_ssim_values, _        = structural_similarity(rgb_image_original, rgb_image_encoded_lsb, full=True, channel_axis=-1)
    y3_steganogan_ssim_values, _ = structural_similarity(rgb_image_original, rgb_image_encoded_steganogan, full=True, channel_axis=-1)
    y4_pvd_ssim_values, _        = structural_similarity(rgb_image_original, rgb_image_encoded_pvd, full=True, channel_axis=-1)

    y2_lsb_values        = np.append(y2_lsb_values, y2_lsb_ssim_values)
    y3_steganogan_values = np.append(y3_steganogan_values, y3_steganogan_ssim_values)
    y4_pvd_values        = np.append(y4_pvd_values, y4_pvd_ssim_values)

    if bpp_value in [0.01, 0.1]:
        filename_encoded_dct  = './encoded_images/{0}/{1}_dct_encoded.png'.format(imtxt, bpp_value)
        image_encoded_dct     = cv2.imread(filename_encoded_dct, 1)
        image_encoded_dct     = cv2.resize(image_encoded_dct, (image_original.shape[1], image_original.shape[0]))
        rgb_image_encoded_dct = cv2.cvtColor(image_encoded_dct, cv2.COLOR_BGR2RGB)
        y1_dct_ssim_values, _ = structural_similarity(rgb_image_original, rgb_image_encoded_dct, full=True, channel_axis=-1)
        y1_dct_values         = np.append(y1_dct_values, y1_dct_ssim_values)
    else:
        y1_dct_values         = np.append(y1_dct_values, 0)

y1_dct_values[2:] = np.nan

# print(x_bpp_values)
# print(y1_dct_values)
# print(y2_lsb_values)
# print(y3_steganogan_values)
# print(y4_pvd_values)
plt.plot(x_bpp_values, y1_dct_values,        marker='o', linestyle='-', label='DCT',        color='blue')
plt.plot(x_bpp_values, y2_lsb_values,        marker='o', linestyle='-', label='LSB',        color='green')
plt.plot(x_bpp_values, y3_steganogan_values, marker='o', linestyle='-', label='STEGANOGAN', color='purple')
plt.plot(x_bpp_values, y4_pvd_values,        marker='o', linestyle='-', label='PVD',        color='red')

# Add labels and a legend
plt.xlabel('BPP values')
plt.ylabel('SSIM')
plt.title('Change of SSIM with BPP')
plt.legend()

# Show the plot
# plt.show()
filename_to_save = "./metric_plots/ssim/ssim_plot.png".format(imtxt, bpp_value)
plt.savefig(filename_to_save)
