import json
import os

import cv2
import pytesseract
from PIL import Image
from pytesseract import Output
from tqdm import tqdm

from utils import config
from utils.files import parse_doc_type_arg, get_data_files
import numpy as np


def get_tesseract_results(image_path):
    img = Image.open(image_path)
    dpi = int(img.info['dpi'][0])

    ocr_result = pytesseract.image_to_data(np.asarray(img), lang='bul+eng', config=f'--oem 1 --psm 11 --dpi {dpi}', output_type=Output.DICT)
    return ocr_result


if __name__ == '__main__':
    os.makedirs(config.TESSERACT_RESULTS_LSTM_DIR, exist_ok=True)

    doc_types = parse_doc_type_arg()
    images = get_data_files(config.DATA_DIR, doc_types=doc_types)

    for image_name, image_path in tqdm(images, desc='Generating Tesseract Results'):
        result = get_tesseract_results(image_path)
        with open(os.path.join(config.TESSERACT_RESULTS_LSTM_DIR, image_name + '.json'), 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
