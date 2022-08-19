import os

import numpy as np
from PIL import Image
from skimage import transform, filters
from skimage.filters.thresholding import threshold_otsu
from skimage.morphology import flood_fill
from skimage.util import crop
from tqdm import tqdm

from utils import config


def remove_black_from_border(img):
    gray = np.asarray(img.convert('L'))
    thresh = threshold_otsu(gray)
    binary = gray > thresh
    padded = np.pad(binary, pad_width=1, mode='constant', constant_values=0)
    filled = flood_fill(padded, (0, 0), 1)
    cropped = crop(filled, 1)
    gray_again = Image.fromarray(cropped)
    rgb_again = gray_again.convert('RGB')
    return rgb_again


def correct_image_skew(img):
    gray = np.asarray(img.convert('L'))

    edges = filters.sobel(gray)

    theta = np.linspace(80 * (np.pi / 180), 100 * (np.pi / 180), 81)
    out, angles, distances = transform.hough_line(edges, theta)
    _, angles_peaks, distances_peaks = transform.hough_line_peaks(
        out, angles, distances, num_peaks=50, threshold=0.05 * np.max(out)
    )
    angles_peaks = np.rad2deg(angles_peaks)
    angles_peaks = 90 - angles_peaks

    vals, cnts = np.unique(angles_peaks, return_counts=True)
    skew = vals[np.argmax(cnts)]

    rotated = transform.rotate(np.asarray(img), -skew, cval=1)
    rgb_again = Image.fromarray((rotated * 255).astype(np.uint8))
    return rgb_again, skew


def show_img(img, cmap='gray', title='title'):
    from matplotlib import pyplot as plt
    plt.figure(figsize=(8, 12))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.show()


def preprocess(img):
    img = remove_black_from_border(img)
    img, skew = correct_image_skew(img)

    img = img.convert('L')
    return img


if __name__ == '__main__':
    os.makedirs(config.DATA_DIR, exist_ok=True)

    for i in tqdm(os.listdir(config.RAW_DATA_DIR)):
        if not os.path.splitext(i)[1]:
            img = Image.open(os.path.join(config.RAW_DATA_DIR, i))
            dpi = img.info['dpi']

            img = preprocess(img)

            img.save(os.path.join(config.DATA_DIR, i + '.png'), dpi=dpi)

    #         img_path = os.path.join(PNG_DATA_DIR, i + '.png')
    #         img = Image.open(img_path)
    #
    # for i in os.listdir(PNG_DATA_DIR):
    #     img_path = os.path.join(PNG_DATA_DIR, i)
    #     img = Image.open(img_path)
    #     print(img.info)
