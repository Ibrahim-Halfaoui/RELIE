import glob
import json
import os

import pandas as pd
from tqdm import tqdm

import utils.config


def iou(boxA, boxB):
    boxA = [boxA['x1'], boxA['y1'], boxA['x2'], boxA['y2']]
    boxB = [boxB['x1'], boxB['y1'], boxB['x2'], boxB['y2']]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    try:
        iou = interArea / boxBArea
    except ZeroDivisionError:
        return None
    # return the intersection over union value
    return iou


if __name__ == '__main__':
    candidate_files = glob.glob(os.path.join(utils.config.CANDIDATES_DIR, '*.json'))
    label_files = [os.path.join(utils.config.LABELS_DIR, os.path.basename(_)) for _ in
                   candidate_files]

    n_cands = 0
    res = []

    for cand_path, label_path in tqdm(list(zip(candidate_files, label_files))):
        cands = json.load(open(cand_path))
        labels = json.load(open(label_path))

        for label in labels['shapes']:
            key = label['label']
            coords = {'x1': label['points'][0][0],
                      'y1': label['points'][0][1],
                      'x2': label['points'][1][0],
                      'y2': label['points'][1][1]}

            key_cands = cands[key]
            ious = [iou(_, coords) for _ in key_cands]
            max_iou = max(ious) if ious else 0

            res_dic = {}
            res_dic['file'] = os.path.splitext(os.path.basename(cand_path))[0]
            res_dic['key'] = key
            res_dic['iou'] = max_iou
            res_dic['n_cands'] = len(key_cands)
            res_dic['missing'] = not ious
            res.append(res_dic)

    df = pd.DataFrame(res)
    print(df.groupby('key')['n_cands', 'missing', 'iou'].mean().sort_values('iou'))

    print(df.iou.mean())

    ndf = df.groupby(['file', 'key']).iou.mean()
    sorted_keys = df.groupby('file').iou.mean().sort_values().index
    with pd.option_context('display.max_rows', 100, 'display.max_columns', 10):
        print(ndf[sorted_keys])
