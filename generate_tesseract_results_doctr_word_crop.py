import json
import os
from collections import defaultdict
from functools import lru_cache
from typing import List

import numpy as np
from PIL import Image, ImageDraw
from doctr.models.detection.zoo import detection_predictor
from doctr.models.predictor.tensorflow import OCRPredictor
from doctr.models.recognition.zoo import recognition_predictor
from pytesseract import Output
from pytesseract import pytesseract
from skimage import morphology
from tqdm import tqdm

from candidates.tesseract_results import Word, Document
from utils import config
from utils.files import parse_doc_type_arg, get_data_files
from utils.viz_utils import show_img, render_words


def get_tesseract_results(img):
    dpi = int(img.info['dpi'][0])

    ocr_result = pytesseract.image_to_data(np.asarray(img), lang='bul+eng', config=f'--oem 1 --psm 11 --dpi {dpi}',
                                           output_type=Output.DICT)
    return ocr_result


@lru_cache
def get_word_detector_doctr():
    det_predictor = detection_predictor('db_resnet50', pretrained=True)
    reco_predictor = recognition_predictor('crnn_vgg16_bn', pretrained=True)
    predictor = OCRPredictor(det_predictor=det_predictor, reco_predictor=reco_predictor,
                             assume_straight_pages=True)
    return predictor


def detect_words_doctr(img):
    predictor = get_word_detector_doctr()
    img = np.asarray(img.convert('RGB'))

    model_output = predictor([img])

    words = []
    for page in model_output.pages:
        for block in page.blocks:
            for line in block.lines:
                for w in line.words:
                    x1 = w.geometry[0][0] * img.shape[1]
                    y1 = w.geometry[0][1] * img.shape[0]
                    x2 = w.geometry[1][0] * img.shape[1]
                    y2 = w.geometry[1][1] * img.shape[0]
                    new_word = Word(x1, y1, x2 - x1, y2 - y1, w.value, w.confidence)
                    words.append(new_word)
    return words


@lru_cache
def get_word_detector_mmocr():
    from mmocr.utils.ocr import MMOCR

    # Load models into memory
    ocr = MMOCR(det='DBPP_r50', recog=None, config_dir=r'/home/stefan.kokov/mmocr/configs')
    return ocr


def detect_words_mmocr(img):
    '''
    Detect words using mmocr DBNET
    @param img: RGB image
    @return: list of detected words
    '''

    ocr = get_word_detector_mmocr()

    open_cv_image = np.asarray(img)
    results = ocr.readtext(open_cv_image)

    words = []
    for bbox in results[0]['boundary_result']:
        bbox = bbox[:-1]
        xmax = int(max(bbox[::2]))
        xmin = int(min(bbox[::2]))
        ymax = int(max(bbox[1::2]))
        ymin = int(min(bbox[1::2]))
        w = Word(xmin, ymin, xmax - xmin, ymax - ymin, None, None)
        words.append(w)
    return words


def delete_words_from_document(img, words):
    mode = img.mode
    img = img.copy().convert('L')
    draw = ImageDraw.Draw(img, 'L')
    for w in words:
        draw.rectangle([w.x1, w.y1, w.x2, w.y2], outline=255, fill=255)
    return img.convert(mode)


def clean_document(img, words):
    new_img = Image.new('RGB', (img.size[0], img.size[1]), (255, 255, 255))
    for w in words:
        crop = img.crop((w.x1, w.y1, w.x2, w.y2))
        new_img.paste(crop, (int(w.x1), int(w.y1)))

    return new_img


def crop_words(img, words):
    crops = []
    for w in words:
        crop = img.crop((w.x1, w.y1, w.x2, w.y2))
        crops.append(crop)
    return crops

def erode_img(img, order):
    img = img.copy()
    img = morphology.erosion(np.asarray(img), np.array([[1] * order]).transpose())
    img = morphology.erosion(np.asarray(img), np.array([[1] * order]))
    img = Image.fromarray((img).astype(np.uint8)).convert('RGB')
    return img


def dilate_img(img, order):
    img = img.copy()
    img = morphology.dilation(np.asarray(img), np.array([[1] * order]).transpose())
    img = morphology.dilation(np.asarray(img), np.array([[1] * order]))
    img = Image.fromarray((img).astype(np.uint8)).convert('RGB')
    return img


if __name__ == '__main__':
    DOCTR_TEXT_DETECTION_IMG = os.path.join(config.BASE_DATA_DIR, 'doctr_text_detection_img')
    os.makedirs(DOCTR_TEXT_DETECTION_IMG, exist_ok=True)
    TEXT_DETECTION_IMG = os.path.join(config.BASE_DATA_DIR, 'text_detection_img')
    os.makedirs(TEXT_DETECTION_IMG, exist_ok=True)
    os.makedirs(config.TESSERACT_RESULTS_LSTM_DIR, exist_ok=True)

    doc_types = parse_doc_type_arg()
    images = get_data_files(config.DATA_DIR, doc_types=doc_types)

    for image_name, image_path in images:
        img = Image.open(image_path).convert('L')
        dpi = img.info['dpi']

        get_word_detector_doctr()

        words1 = detect_words_doctr(img.convert('RGB'))

        res = delete_words_from_document(img.convert('RGB'), words1)

        words2 = detect_words_mmocr(res.convert('RGB'))

        words = words1 + words2

        clean_img = clean_document(img, words)
        clean_img.info['dpi'] = dpi

        new_words = []
        ocr_results = defaultdict(list)
        word_crops = crop_words(img, words)
        for w in tqdm(words, desc=f'OCR of {image_name}'):
            crop = img.crop((w.x1, w.y1, w.x2, w.y2))
            ocr_result = pytesseract.image_to_data(np.asarray(crop), lang='bul+eng',
                                                   config=f'--oem 1 --psm 8 --dpi {dpi}',
                                                   output_type=Output.DICT)

            ocr_result['top'] = [w.top] * len(ocr_result['top'])
            ocr_result['left'] = [w.left] * len(ocr_result['left'])
            ocr_result['width'] = [w.width] * len(ocr_result['width'])
            ocr_result['height'] = [w.height] * len(ocr_result['height'])
            for key, val in ocr_result.items():
                ocr_results[key] = ocr_results[key] + val


        with open(os.path.join(config.TESSERACT_RESULTS_LSTM_DIR, image_name + '.json'), 'w', encoding='utf-8') as f:
            json.dump(ocr_results, f, ensure_ascii=False)
        #     new_words.append(Word(w.left, w.top, w.width, w.height, txt, conf))
        #
        # rend_img = render_words(img, new_words)
        # show_img(rend_img)