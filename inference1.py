import json
import os
import pickle
import traceback
from collections import defaultdict
from candidates.tesseract_results import Word
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import utils.config
from utils import Neighbour, config, preprocess
from utils.labels import get_annotations
from utils.neighbours_new import attach_neighbours
from vizualize_candidates import draw_box, draw_legend
import argparse

def attach_neighbour_candidates(width, height, ocr_data, candidates):
    empty_index = [i for i, ele in enumerate(ocr_data['text']) if ele == ""]
    for key in ocr_data.keys():
        ocr_data[key] = [j for i, j in enumerate(ocr_data[key]) if i not in empty_index]

    cand_words = [cad for both_cads in candidates.values() for cad in both_cads] + \
                 [cad for both_cads in candidates.values() for cad in both_cads]
    cand_words = [Word(cand['x1'],
                       cand['y1'],
                       cand['x2'] - cand['x1'],
                       cand['y2'] - cand['y1'],
                       cand['text'] if 'text' in cand else '',
                       None) for cand in cand_words]
    words = []
    for txt, x, y, w, h in zip(ocr_data['text'], ocr_data['left'], ocr_data['top'], ocr_data['width'],
                               ocr_data['height']):

        word = Word(x, y, w, h, txt, None)
        if any([w1.contains(word) for w1 in cand_words]):
            continue

        words.append(word.to_dict())

    x_offset = int(width * 0.1)
    y_offset = int(height * 0.1)
    for cls, both_cads in candidates.items():
        for cad in both_cads:
            neighbours = Neighbour.find_neighbour(cad, words, x_offset, y_offset, width, height)
            cad['neighbours'] = neighbours
    return candidates


def load_saved_vocab(path):
    cached_data = pickle.load(open(path, 'rb'))
    return cached_data['vocab'], cached_data['mapping']


def parse_input(annotations, fields_dict, n_neighbours=5, vocabulary=None):
    """Generates input samples from annotations data."""
    field_ids = list()
    candidate_cords = list()
    neighbours = list()
    neighbour_cords = list()
    n_classes = len(fields_dict)
    for field, value in annotations.items():
        if annotations[field]:
            for idx, val in enumerate(value):
                _neighbours, _neighbour_cords = preprocess.get_neighbours(
                    val['neighbours'],
                    vocabulary, n_neighbours
                )
                field_ids.append(np.eye(n_classes)[fields_dict[field]])
                candidate_cords.append(
                    [
                        val['x'],
                        val['y']
                    ]
                )
                neighbours.append(_neighbours)
                neighbour_cords.append(_neighbour_cords)
    return torch.Tensor(field_ids).type(torch.FloatTensor), torch.Tensor(candidate_cords).type(
        torch.FloatTensor), torch.Tensor(neighbours).type(torch.int64), torch.Tensor(neighbour_cords).type(
        torch.FloatTensor)


def normalize_coordinates(annotations, width, height):
    try:
        for cls, cads in annotations.items():
            for i, cd in enumerate(cads):
                cd = cd.copy()
                x1 = cd['x1']
                y1 = cd['y1']
                x2 = cd['x2']
                y2 = cd['y2']
                cd['x'] = ((x1 + x2) / 2) / width
                cd['y'] = ((y1 + y2) / 2) / height
                neighbours = []
                for neh in cd['neighbours']:
                    neh = neh.copy()
                    x1_neh = neh['x1']
                    y1_neh = neh['y1']
                    x2_neh = neh['x2']
                    y2_neh = neh['y2']
                    # calculating neighbour position w.r.t candidate
                    neh['x'] = (((x1_neh + x2_neh) / 2) / width) - cd['x']
                    neh['y'] = (((y1_neh + y2_neh) / 2) / height) - cd['y']
                    neighbours.append(neh)
                cd['neighbours'] = neighbours
                annotations[cls][i] = cd
    except Exception:
        trace = traceback.format_exc()
        print("Error in normalizing position: %s : %s" % (trace, trace))
    return annotations


def predict(image_path, vocab_path, model_path, cuda=True):
    device = torch.device('cuda:0' if cuda else 'cpu')

    # image = cv2.imread(args.image_path)
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    # ocr_results = generate_tesseract_results.get_tesseract_results(args.image_path)
    # ocr_results = generate_tesseract_results.get_tesseract_results(image_path)
    cand_path = os.path.join(utils.config.CANDIDATES_DIR, os.path.splitext(os.path.basename(image_path))[0] + '.json')
    ocr_path = os.path.join(utils.config.TESSERACT_RESULTS_LSTM_DIR,
                            os.path.splitext(os.path.basename(image_path))[0] + '.json')
    ocr_results = json.load(open(ocr_path))
    candidates = json.load(open(cand_path))

    vocab, class_mapping = load_saved_vocab(vocab_path)
    # candidates = extract_candidates.get_candidates(cand_path)

    candidates_with_neighbours = attach_neighbour_candidates(width, height, ocr_results, candidates)
    # candidates_with_neighbours = attach_neighbours(candidates, ocr_results, n_neighbours=config.NEIGHBOURS)
    annotation = normalize_coordinates(candidates_with_neighbours, width, height)
    _data = parse_input(annotation, class_mapping, config.NEIGHBOURS, vocab)
    field_ids, candidate_cords, neighbours, neighbour_cords = _data
    rlie = torch.load(model_path)
    rlie = rlie.to(device)
    field_ids = field_ids.to(device)
    candidate_cords = candidate_cords.to(device)
    neighbours = neighbours.to(device)
    neighbour_cords = neighbour_cords.to(device)
    field_idx_candidate = np.argmax(field_ids.detach().to('cpu').numpy(), axis=1)
    with torch.no_grad():
        rlie.eval()
        val_outputs = rlie(field_ids, candidate_cords, neighbours, neighbour_cords, masks=None)
    val_outputs = val_outputs.to('cpu').numpy()
    out = {cl: val_outputs[np.where(field_idx_candidate == cl)] for cl in np.unique(field_idx_candidate)}

    new_cands = defaultdict(list)
    for idx, (key, value) in enumerate(candidates.items()):
        if value:
            probas = out[class_mapping[key]]
            for v, p in zip(value, probas):
                dic = {
                    'text': v['text'],
                    'x1': v['x1'],
                    'y1': v['y1'],
                    'x2': v['x2'],
                    'y2': v['y2'],
                    'proba': float(p)
                }
                new_cands[key].append(dic)
    return new_cands

# def main():
#     cuda = True
#     load_model = './output/model.pth'
#     saved_path = './output/cached_data_train.pickle'
#     image_path = ['/data/uniqa_samples/data/{51B82D46-FEC0-4A2A-87E8-CDC689AA95D8}.png',
#                   '/data/uniqa_samples/data/{A47553EA-3551-4137-814F-A7CFFCA5C5F6}.png',
#                   '/data/uniqa_samples/data/{B8A87080-CB59-4426-A5AA-72A71DB1E6C0}.png'][0]
#     output_candidates = predict(image_path, saved_path, load_model, cuda)
#
#     candidates_path = os.path.join(utils.config.CANDIDATES_DIR, os.path.splitext(os.path.basename(image_path))[0] + '.json')
#     candidates = json.load(open(candidates_path))
#
#     labels_path = os.path.join(utils.config.LABELS_DIR, os.path.splitext(os.path.basename(image_path))[0] + '.json')
#     labels = json.load(open(labels_path))
#     labels = {lbl['label']: {'x1': lbl['points'][0][0],
#                              'y1': lbl['points'][0][1],
#                              'x2': lbl['points'][1][0],
#                              'y2': lbl['points'][1][1]
#                              } for lbl in labels['shapes']}
#
#     font = ImageFont.truetype(utils.config.CYRILLIC_FONT_FILE, 50)
#     img = Image.open(image_path).convert('RGB')
#     draw = ImageDraw.Draw(img)
#     for k, v in labels.items():
#         draw.rectangle(((v['x1'], v['y1']), (v['x2'], v['y2'])), outline=(255, 0, 0), width=3)
#         draw.text((v['x1'], v['y1']), k, fill=(0, 0, 255), anchor='lb', font=font)
#     for k, l in candidates.items():
#         for v in l:
#             draw.rectangle(((v['x1'], v['y1']), (v['x2'], v['y2'])), outline=(0, 0, 255), width=3)
#             draw.text((v['x1'], v['y1']), k, fill=(0, 0, 255), anchor='lb', font=font)
#     for k, v in output_candidates.items():
#         draw.rectangle(((v['x1'], v['y1']), (v['x2'], v['y2'])), outline=(0, 255, 0), width=3)
#         draw.text((v['x1'], v['y1']), k, fill=(0, 0, 255), anchor='lb', font=font)
#
#     from matplotlib import pyplot as plt
#
#     plt.figure(figsize=(15, 15))
#     plt.imshow(img)
#     plt.show()
#
#     print(output_candidates)

########################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference code for Relie Model.')
    parser.add_argument('--doc_type', type=str, help='Doc type',
                        default=None)
    parser.add_argument('--image_paths_file', type=str, help='list with Paths to images',
                        default='inf_img_list.txt')
    parser.add_argument('--cuda', type=bool, help='Cuda Falg',
                        default=True)
    args = parser.parse_args()
    cuda = args.cuda
    if args.doc_type:
        load_model = './output/model_' + args.doc_type + '.pth'
        saved_path = './output/cached_data_train_' + args.doc_type + '.pickle'
    else:
        print('PLEASE DEFINE DOC_TYPE IN ARGUMENT LIST')
        quit()
    # with open(args.image_paths_file) as f:
    #     image_paths = f.readlines()

    # Uncomment to Use filtered images based on doc type
    annot = get_annotations()
    image_paths = [os.path.join(utils.config.DATA_DIR, i)
                   for i in os.listdir(utils.config.DATA_DIR)
                   if
                   annot[os.path.splitext(i)[0]]['flags'][args.doc_type]
                   ]

    for image_path in tqdm(image_paths):
        try:
            image_path = image_path.strip()
            output_candidates = predict(image_path, saved_path, load_model, cuda)

            labels_path = os.path.join(utils.config.LABELS_DIR, os.path.splitext(os.path.basename(image_path))[0] + '.json')
            lbls = defaultdict(list)
            labels = json.load(open(labels_path))
            for lbl in labels['shapes']:
                lbls[lbl['label']].append({'x1': lbl['points'][0][0],
                                     'y1': lbl['points'][0][1],
                                     'x2': lbl['points'][1][0],
                                     'y2': lbl['points'][1][1]
                                     })
            labels = lbls
            cands = output_candidates
            colors = {'candidate': (0, 0, 255),
                      'prediction': (0, 255, 0),
                      'ground_truth': (255, 0, 0)}

            for key, value in cands.items():
                img = Image.open(image_path).convert('RGB')
                value = sorted(value, key=lambda x: x['proba'], reverse=True)
                num_items = len(labels[key])
                for i, v in enumerate(value):
                    color = colors['prediction'] if i < num_items else colors['candidate']
                    img = draw_box(img, v['x1'], v['y1'], v['x2'], v['y2'], color=color, text='{:.2f}'.format(v['proba']))

                for v in labels[key]:
                    img = draw_box(img, v['x1'], v['y1'], v['x2'], v['y2'], color=colors['ground_truth'])

                img = draw_legend(img, colors)
                os.makedirs(utils.config.PREDICTION_DIR_IMG, exist_ok=True)
                out_path = os.path.join(utils.config.PREDICTION_DIR_IMG, os.path.splitext(os.path.basename(image_path))[0] + f'_{key}.png')
                img.save(out_path)
                # plt.figure(figsize=(12, 12))
                # plt.imshow(img)
                # plt.show()
        except:
            traceback.print_exc()
