import json
import os
import re

from PIL import Image
from dateparser.search import search_dates
from tqdm import tqdm

import utils.config
from candidates import ambolatory, invoice, receipt
from utils import config
from utils.files import parse_doc_type_arg, get_data_files
from utils.labels import get_annotations


def get_invoice_nums(all_words):
    inv_nums = []
    invoice_no_re = r'^[0-9a-zA-Z-:]+$'
    for word in all_words:
        if not re.search('\d', word['text']):
            continue
        if len(word['text']) < 3:
            continue
        result = re.findall(invoice_no_re,word['text'])
        if result:
            inv_nums.append({
                'text': word['text'],
                'x1': word['left'],
                'y1': word['top'],
                'x2': word['left'] + word['width'],
                'y2': word['top'] + word['height']
            })

    return inv_nums


def get_dates(all_text, all_words):
    dates, all_dates = [], []
    indices = []
    # matches = search_dates(all_text)
    parsers = ['absolute-time', 'custom-formats']
    matches = search_dates(all_text, settings={'DATE_ORDER': 'DMY', 'STRICT_PARSING': True, 'PARSERS': parsers})

    for match in matches:
        text = match[0]

        token_length = len(text.split(' '))
        idx = all_text.find(match[0])
        text_len = len(text)
        index = len(all_text[:idx].strip().split(' '))

        replaced_text = ' '.join(['*' * len(i) for i in text.split(' ')])

        indices.append(list(range(index, index + token_length)))

        index += token_length
        all_text = all_text[:idx + text_len].replace(text, replaced_text) + all_text[idx + text_len:]

    for date_indices in indices:
        date = ''
        left, top, right, bottom = [], [], [], []
        for i in date_indices:
            date += ' ' + all_words[i]['text']
            left.append(all_words[i]['left'])
            top.append(all_words[i]['top'])
            right.append(all_words[i]['left'] + all_words[i]['width'])
            bottom.append(all_words[i]['top'] + all_words[i]['height'])
        all_dates.append({
            'text': date.strip(),
            'x1': min(left),
            'y1': min(top),
            'x2': max(right),
            'y2': max(bottom)
        })

    return all_dates


def get_amounts(all_words):
    amounts = []
    # amount_re = r"\$?([0-9]*,)*[0-9]{3,}(\.[0-9]+)?"
    amount_re = r"\$?^\d*\.?\d+$"
    for word in all_words:
        if not re.search(amount_re, word['text']):
            continue
        try:
            formatted_word = re.sub(r'[$,]','', word['text'])
            float(formatted_word)
        
            amounts.append({
                'text': word['text'],
                'x1': word['left'],
                'y1': word['top'],
                'x2': word['left'] + word['width'],
                'y2': word['top'] + word['height']
            })

        except ValueError:
            continue

    return amounts


def get_candidates(img, tesseract_results):
    annotations = get_annotations()
    f_name = os.path.splitext(os.path.split(file_name)[-1])[0]

    if annotations[f_name]['flags']['ambolatory']:
        cand = ambolatory.get_candidates(img, tesseract_results)
    elif annotations[f_name]['flags']['invoice']:
        cand = invoice.get_candidates(img, tesseract_results)
    elif annotations[f_name]['flags']['receipt']:
        cand = receipt.get_candidates(img, tesseract_results)
    else:
        cand = None

    return cand


if __name__ == "__main__":
    os.makedirs(utils.config.CANDIDATES_DIR, exist_ok=True)

    doc_types = parse_doc_type_arg()
    files_list = get_data_files(config.TESSERACT_RESULTS_LSTM_DIR, doc_types=doc_types)

    for file_name, file_path in tqdm(files_list, desc='Generating candidates'):
        img = Image.open(os.path.join(config.DATA_DIR, file_name + '.png'))
        tesseract_result = json.load(open(file_path))
        cand = get_candidates(img, tesseract_result)
        if cand:
            with open(os.path.join(utils.config.CANDIDATES_DIR, file_name + '.json'), 'w', encoding='utf-8') as f:
                json.dump(cand, f, ensure_ascii=False)
