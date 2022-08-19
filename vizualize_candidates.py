import json
import os

import seaborn as sns
from PIL import ImageFont, ImageDraw, Image
from tqdm import tqdm

from utils import config
from utils.files import parse_doc_type_arg, get_data_files, get_doc_type


def draw_box(image, x1, y1, x2, y2, text=None, color=(255, 255, 255)):
    draw = ImageDraw.Draw(image)

    draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=5)
    font = ImageFont.truetype('/home/stefan.kokov/temp/Alice-Regular.ttf', 42)
    if text:
        draw.text((x1, y1), text, language='bg', font=font, fill=color, anchor='lb')

    return image


def draw_legend(image, color_map):
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype('/home/stefan.kokov/temp/Alice-Regular.ttf', 42)
    for i, (k, v) in enumerate(color_map.items()):
        draw.text((0, i * 42), k, language='bg', font=font, fill=v, anchor='lt')

    return image


def get_key_colors(keys, palette='bright'):
    num_colors = len(keys)
    colors = sns.color_palette(palette, num_colors)
    return {key: tuple([int(_ * 255) for _ in col]) for key, col in zip(keys, colors)}


def main():
    os.makedirs(config.CANDIDATES_DIR_IMG, exist_ok=True)

    doc_types = parse_doc_type_arg()
    candidate_files = get_data_files(config.CANDIDATES_DIR, doc_types=doc_types)

    for (cand_name, cand_path) in tqdm(candidate_files):
        img_path = os.path.join(config.DATA_DIR, cand_name + '.png')

        ann = get_doc_type(cand_name)

        img = Image.open(img_path).convert('RGB')
        cands = json.load(open(cand_path))
        colors = get_key_colors(list(set(cands.keys())))
        offset = {key: i * 5 for i, key in enumerate(colors.keys())}

        for key, value in cands.items():
            for v in value:
                ofs = offset[key]
                img = draw_box(img, v['x1'] - ofs, v['y1'] - ofs, v['x2'] + ofs, v['y2'] + ofs, color=colors[key])

        img = draw_legend(img, colors)
        if ann:
            out_file = os.path.join(config.CANDIDATES_DIR_IMG, cand_name + '_' + ann + '.png')
        else:
            out_file = os.path.join(config.CANDIDATES_DIR_IMG, cand_name + '.png')

        img.save(out_file)


if __name__ == '__main__':
    main()
