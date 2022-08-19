import json
import os

from PIL import Image
from tqdm import tqdm

from candidates.tesseract_results import Document
from utils import config
from utils.files import parse_doc_type_arg, get_data_files, get_doc_type

if __name__ == '__main__':
    os.makedirs(config.TESSERACT_RESULTS_LSTM_IMG_DIR, exist_ok=True)

    doc_types = parse_doc_type_arg()
    tesseract_results_files = get_data_files(config.TESSERACT_RESULTS_LSTM_DIR, doc_types=doc_types)

    for file_name, file_path in tqdm(tesseract_results_files, desc='Visualizing tesseract results'):
        ann = get_doc_type(file_name)

        img = Image.open(os.path.join(config.DATA_DIR, file_name + '.png'))
        tesseract_result = json.load(open(file_path))

        doc = Document(img, tesseract_result)
        doc.parse_horizontal_lines()
        doc.parse_vertical_lines()
        res_img = doc.render_document(words=True, lines=False, paragraphs=False, blocks=False, horizontal_lines=True,
                                      vertical_lines=True)

        if not ann:
            out_file = os.path.join(config.TESSERACT_RESULTS_LSTM_IMG_DIR, file_name + '.png')
        else:
            out_file = os.path.join(config.TESSERACT_RESULTS_LSTM_IMG_DIR, file_name + '_' + ann + '.png')
        res_img.save(out_file)
