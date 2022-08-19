import argparse
import json
import os
import pickle
import traceback
from itertools import chain

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import utils.config
from utils import Neighbour, config, preprocess


def attach_neighbour_candidates(width, height, ocr_data, candidates):
    empty_index = [i for i, ele in enumerate(ocr_data['text']) if ele == ""]
    for key in ocr_data.keys():
        ocr_data[key] = [j for i, j in enumerate(ocr_data[key]) if i not in empty_index]
    words = []
    for txt, x, y, w, h in zip(ocr_data['text'], ocr_data['left'], ocr_data['top'], ocr_data['width'],
                               ocr_data['height']):
        x2 = x + w
        y2 = y + h
        words.append({'text': txt, 'x1': x, 'y1': y, 'x2': x2, 'y2': y2})
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
    out = {cl: np.argmax(val_outputs[np.where(field_idx_candidate == cl)]) for cl in np.unique(field_idx_candidate)}
    print(val_outputs)
    all_cands = list(chain.from_iterable([j for i, j in candidates.items()]))
    for cand, prob in zip(all_cands, val_outputs):
        print(prob, cand)
    # print(all_cands)
    output_candidates = {}
    for idx, (key, value) in enumerate(candidates.items()):
        # idx = 2
        if idx in out:
            candidate_idx = out[idx]
            # cand = candidates[key][candidate_idx]
            for i0 in range(len(value)):
                cand = candidates[key][i0]
                cand_coords = [cand['x1'], cand['y1'], cand['x2'], cand['y2']]
                if i0 == candidate_idx:
                    output_candidates[key] = {'text': cand['text'], 'x1': cand['x1'], 'y1': cand['y1'],
                                              'x2': cand['x2'], 'y2': cand['y2']}
                    true_candidate_color = (0, 255, 0)
                else:
                    true_candidate_color = (0, 0, 255)
                # cv2.rectangle(output_image, (cand_coords[0], cand_coords[1]), (cand_coords[2], cand_coords[3]),
                #               true_candidate_color, 2)
    return output_candidates

###########################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference code for Relie Model.')
    parser.add_argument('--doc_type', type=str, help='Doc type',
                        default=None)
    parser.add_argument('--image_paths_file',type=str, help='list with Paths to images',
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

    with open(args.image_paths_file) as f:
        image_paths = f.readlines()

    # # Uncomment to Use filtered images based on doc type
    # annot = get_annotations()
    # image_paths = [os.path.join(utils.config.DATA_DIR, i)
    #                for i in os.listdir(utils.config.DATA_DIR)
    #                if
    #                annot[os.path.splitext(i)[0]]['flags'][args.doc_type]
    #                ]

    for line in tqdm(image_paths):
            line = line.strip()
            image_path = [line][0]
            output_candidates = predict(image_path, saved_path, load_model, cuda)

            candidates_path = os.path.join(utils.config.CANDIDATES_DIR, os.path.splitext(os.path.basename(image_path))[0] + '.json')
            candidates = json.load(open(candidates_path))

            labels_path = os.path.join(utils.config.LABELS_DIR, os.path.splitext(os.path.basename(image_path))[0] + '.json')
            labels = json.load(open(labels_path))
            labels = {lbl['label']: {'x1': lbl['points'][0][0],
                                     'y1': lbl['points'][0][1],
                                     'x2': lbl['points'][1][0],
                                     'y2': lbl['points'][1][1]
                                     } for lbl in labels['shapes']}

            font = ImageFont.truetype(utils.config.CYRILLIC_FONT_FILE, 50)
            img = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(img)
            for k, v in labels.items():
                draw.rectangle(((v['x1'], v['y1']), (v['x2'], v['y2'])), outline=(255, 0, 0), width=3)
                draw.text((v['x1'], v['y1']), k, fill=(0, 0, 255), anchor='lb', font=font)
            for k, l in candidates.items():
                for v in l:
                    draw.rectangle(((v['x1'], v['y1']), (v['x2'], v['y2'])), outline=(0, 0, 255), width=3)
                    draw.text((v['x1'], v['y1']), k, fill=(0, 0, 255), anchor='lb', font=font)
            for k, v in output_candidates.items():
                draw.rectangle(((v['x1'], v['y1']), (v['x2'], v['y2'])), outline=(0, 255, 0), width=3)
                draw.text((v['x1'], v['y1']), k, fill=(0, 0, 255), anchor='lb', font=font)

            from matplotlib import pyplot as plt

            plt.figure(figsize=(15, 15))
            plt.imshow(img)
            plt.show()

            print(output_candidates)
##########################################################################################################################
# def parse_args():
#     """
#     Parse input arguments
#     """
#     parser = argparse.ArgumentParser(description='Inference outputs')
#     parser.add_argument('--cached_pickle', dest='saved_path',
#                         help='Enter the path of the saved pickle during training',
#                         default='cached_data.pickle', type=str)
#     parser.add_argument('--load_saved_model', dest='load_model',
#                         help='directory to load models', default="model.pth",
#                         type=str)
#     parser.add_argument('--image', dest='image_path',
#                         help='directory to load models',
#                         type=str)
#     parser.add_argument('--visualize', dest='visualize',
#                         help='directory to load models',
#                         action='store_true')
#     parser.add_argument('--cuda', dest='cuda',
#                         help='whether use CUDA',
#                         action='store_true')
#     args = parser.parse_args()
#     return args


# def main():
#     cuda = True
#     load_model = './output/model_ambolatory.pth'
#     saved_path = './output/cached_data_train_ambolatory.pickle'
#     visualize = True
#     # args = parse_args()
#     # if torch.cuda.is_available() and not args.cuda:
#     #     print("WARNING: You have a CUDA device, so you should probably run with --cuda")
#     # if not os.path.exists(args.image_path):
#     #     raise Exception("Image not found")
#     device = torch.device('cuda:0' if cuda else 'cpu')
#     dataset_dir = '/home/ibrahim.halfaoui/projects/relie/dataset'
#     images_dir = config.IMAGE_DIR
#     images = ['/data/uniqa_samples/data/{51B82D46-FEC0-4A2A-87E8-CDC689AA95D8}.png',
#               '/data/uniqa_samples/data/{A47553EA-3551-4137-814F-A7CFFCA5C5F6}.png',
#               '/data/uniqa_samples/data/{B8A87080-CB59-4426-A5AA-72A71DB1E6C0}.png']
#
#     for image_path in tqdm(images, desc='Generating inference Results'):
#         # image = cv2.imread(args.image_path)
#         image = cv2.imread(image_path)
#         height, width, _ = image.shape
#         # ocr_results = generate_tesseract_results.get_tesseract_results(args.image_path)
#         # ocr_results = generate_tesseract_results.get_tesseract_results(image_path)
#         cand_path = os.path.join(utils.config.CANDIDATES_DIR, os.path.splitext(os.path.basename(image_path))[0] + '.json')
#         ocr_path = os.path.join(utils.config.TESSERACT_RESULTS_LSTM_DIR,
#                                 os.path.splitext(os.path.basename(image_path))[0] + '.json')
#         ocr_results = json.load(open(ocr_path))
#         candidates = json.load(open(cand_path))
#
#         vocab, class_mapping = load_saved_vocab(saved_path)
#         # candidates = extract_candidates.get_candidates(cand_path)
#
#         candidates_with_neighbours = attach_neighbour_candidates(width, height, ocr_results, candidates)
#         annotation = normalize_coordinates(candidates_with_neighbours, width, height)
#         _data = parse_input(annotation, class_mapping, config.NEIGHBOURS, vocab)
#         field_ids, candidate_cords, neighbours, neighbour_cords = _data
#         rlie = torch.load(load_model)
#         rlie = rlie.to(device)
#         field_ids = field_ids.to(device)
#         candidate_cords = candidate_cords.to(device)
#         neighbours = neighbours.to(device)
#         neighbour_cords = neighbour_cords.to(device)
#         field_idx_candidate = np.argmax(field_ids.detach().to('cpu').numpy(), axis=1)
#         with torch.no_grad():
#             rlie.eval()
#             val_outputs = rlie(field_ids, candidate_cords, neighbours, neighbour_cords, masks=None)
#         val_outputs = val_outputs.to('cpu').numpy()
#         out = {cl: np.argmax(val_outputs[np.where(field_idx_candidate == cl)]) for cl in np.unique(field_idx_candidate)}
#
#         # true_candidate_color = (255, 0, 0)
#         output_candidates = {}
#         output_image = image.copy()
#         for idx, (key, value) in enumerate(candidates.items()):
#             idx = 2
#             if idx in out:
#                 candidate_idx = out[idx]
#                 # cand = candidates[key][candidate_idx]
#                 for i0 in range(len(value)):
#                     cand = candidates[key][i0]
#                     output_candidates[key] = cand['text']
#                     cand_coords = [cand['x1'], cand['y1'], cand['x2'], cand['y2']]
#                     if i0 == candidate_idx:
#                         true_candidate_color = (0, 255, 0)
#                     else:
#                         true_candidate_color = (0, 0, 255)
#                     cv2.rectangle(output_image, (cand_coords[0], cand_coords[1]), (cand_coords[2], cand_coords[3]),
#                                   true_candidate_color, 2)
#         if visualize:
#             from matplotlib import pyplot as plt
#             plt.figure(figsize=(15, 15))
#             plt.imshow(output_image)
#             plt.show()
#             # output_image = cv2.resize(output_image, (int(output_image.shape[1] / 2), int(output_image.shape[0] / 2)),
#             #                           interpolation=cv2.INTER_AREA)
#             # cv2.imshow('Visualize', output_image)
#             # cv2.waitKey()
#     return output_candidates


