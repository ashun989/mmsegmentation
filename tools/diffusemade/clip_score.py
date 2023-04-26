import argparse
import json
import os
import os.path as osp
from abc import ABC, abstractmethod

import clip
import cv2
import nltk
import numpy as np
import torch
import torchvision.transforms.functional
from PIL import Image
from tqdm import tqdm

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

from gen_label_and_prob import read_gray
from mmseg.datasets import PascalVOCDataset

VOC_SYNONYMS = (
    [],
    ['aeroplanes', 'seaplane'],
    ['bicycles'],
    ['birds'],
    ['boats', 'rowboat', 'sailboat', 'motorboat', 'speedboat', 'houseboat'],
    ['bottles'],
    ['buses', 'minibus'],
    ['cars', 'sidecar'],
    ['cats'],
    ['chairs'],
    ['cows'],
    ['diningtables', 'table'],
    ['dogs', 'sheepdog'],
    ['horses', ],
    ['motorbikes', ],
    ['people', ],
    ['pottedplants', 'plant'],
    [],  # sheep
    ['sofas', ],
    ['trains', ],
    ['tvmonitors', 'tv-monitor', 'monitor']
)

VOC_2GRAM = (
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    ['dining table', 'dining tables'],
    [],
    [],
    [],
    [],
    ['potted plant', 'potted plants'],
    [],  # sheep
    [],
    [],
    ['tv monitor', 'tv monitors', 'television monitor']
)

VOC_BACKGROUND_CATEGORY = ['ground', 'land', 'grass', 'tree', 'building', 'wall', 'sky', 'lake', 'water', 'river',
                           'sea', 'railway', 'railroad', 'keyboard', 'helmet',
                           'cloud', 'house', 'mountain', 'ocean', 'road', 'rock', 'street', 'valley', 'bridge', 'sign',
                           ]

VOC_NEW = ['aeroplane', 'bicycle', 'bird avian', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair seat', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person with clothes,people,human',
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor screen',
           ]

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


def one2two_gram(words):
    words2 = []
    for i in range(len(words) - 1):
        words2.append(' '.join([words[i], words[i + 1]]))
    return words2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True, help='Path to img dir')
    parser.add_argument('--ann', type=str, help='Path to ann dir')
    parser.add_argument('--data-info', type=str, required=True, help='Path to data infos')
    parser.add_argument('--img-suffix', type=str, default='.png')
    parser.add_argument('--ann-suffix', type=str, default='.png')
    parser.add_argument('--out', type=str, required=True, help='dir to output files')
    parser.add_argument('--text-method', type=str, default='fix', choices=['ori', 'fix'])
    parser.add_argument('--img-method', type=str, default='whole', choices=['whole', 'boxes'])
    # parser.add_argument('--topk', type=int, default=10)
    return parser.parse_args()


def entity_replace(prompt, tgt_names, cls_one_grams, cls_two_grams):
    prompt_words = nltk.word_tokenize(prompt)  # one-gram
    replaced = []
    for cname in cls_one_grams:
        try:
            idx = prompt_words.index(cname)
            for i in range(len(tgt_names)):
                # if i != cid:
                prompt_words[idx] = tgt_names[i]
                # else:
                #     prompt_words[idx] = cname
                replaced.append(' '.join(prompt_words))
            break
        except ValueError:
            pass
    if replaced:
        return replaced
    prompt_words2 = one2two_gram(prompt_words)
    for cname in cls_two_grams:
        try:
            idx = prompt_words2.index(cname)
            for i in range(len(tgt_names)):
                prompt_words2[idx] = tgt_names[i]
                text = prompt_words2[0]
                for word2 in prompt_words2[1:]:
                    text += word2.split(' ')[-1]
                replaced.append(text)
        except ValueError:
            pass
    return replaced


def zeroshot_classifier(model, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).cuda()
    return zeroshot_weights  # (C, D)


class TopkClassIO:
    def __init__(self, filename):
        self.filename = filename
        self.first_write = True

    def write_line(self, name, labels, probs):
        write_mode = 'a'
        if self.first_write:
            self.first_write = False
            write_mode = 'w'
        with open(self.filename, write_mode) as fp:
            json.dump(dict(name=name, labels=labels, probs=probs), fp)
            fp.write("\n")

    def read_all(self):
        results = {}
        with open(self.filename, 'r') as fp:
            while True:
                line = fp.readline()
                if not line:
                    break
                tmp_dict = json.loads(line.strip())
                name = tmp_dict.pop('name')
                results[name] = tmp_dict
        return results

    # def close(self):
    #     self.fp.close()


class TextFeatureBase(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def get_text_feature(self, *data_args, **data_kwargs):
        pass


def ann2box(ann, multi_contour_eval=True):
    height, width = ann.shape
    ann[ann > 0] = 255
    contours = cv2.findContours(
        image=ann,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]
    if len(contours) == 0:
        return [[0, 0, 0, 0]]
    if not multi_contour_eval:
        contours = [max(contours, key=cv2.contourArea)]
    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])
    return estimated_boxes


def box_area(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def is_box_covered(box1, box2):
    # Judge whether box1 is covered by box2
    if box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]:
        return True
    else:
        return False


def nms_boxes(boxes, min_area=0):
    sorted_boxes = list(zip(boxes, [box_area(box) for box in boxes]))
    sorted_boxes = sorted(sorted_boxes, key=lambda b: b[1], reverse=True)
    if min_area > sorted_boxes[-1][1]:
        i = 0
        while min_area <= sorted_boxes[i][1]:
            i += 1
        sorted_boxes = sorted_boxes[:i]

    sorted_boxes = list(zip(*sorted_boxes))[0]
    removed = [False] * len(sorted_boxes)
    maximum = 0
    while maximum < len(sorted_boxes):
        if not removed[maximum]:
            for j in range(maximum + 1, len(sorted_boxes)):
                if not removed[j] and is_box_covered(sorted_boxes[j], sorted_boxes[maximum]):
                    removed[j] = True
        maximum += 1
    rtn_boxes = [box for i, box in enumerate(sorted_boxes) if not removed[i]]
    return rtn_boxes


class ImageFeatureBase(ABC):
    def __init__(self, model, preprocess):
        self.model = model
        self.preprocess = preprocess

    @abstractmethod
    def get_img_feature(self, img_path, ann_path):
        pass


class ImageFeatureWhole(ImageFeatureBase):
    def __init__(self, model, preprocess):
        super(ImageFeatureWhole, self).__init__(model, preprocess)

    def get_img_feature(self, img_path, ann_path):
        img = Image.open(img_path).convert("RGB")
        img = self.preprocess(img)
        image_input = torch.tensor(np.stack([img])).cuda()
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features  # (1, D)


class ImageFeatureBoxes(ImageFeatureBase):
    def __init__(self, model, preprocess):
        super(ImageFeatureBoxes, self).__init__(model, preprocess)

    def get_img_feature(self, img_path, ann_path):
        assert ann_path is not None and osp.isfile(ann_path), f"No such file: {ann_path}"
        img = Image.open(img_path).convert("RGB")
        ann = read_gray(ann_path)
        boxes = ann2box(ann, multi_contour_eval=True)
        boxes = nms_boxes(boxes, min_area=16 ** 2)
        if not boxes:
            return None
        imgs = []
        for b in boxes:
            crop_img = torchvision.transforms.functional.crop(
                img, b[1], b[0], b[3] - b[1], b[2] - b[0])
            imgs.append(self.preprocess(crop_img))
        image_input = torch.tensor(np.stack(imgs)).cuda()
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features  # (N, D)


class TextFeatureByOrigin(TextFeatureBase):
    def __init__(self, model, out_dir):
        super(TextFeatureByOrigin, self).__init__(model)
        self.undeal_path = osp.join(out_dir, "undeal.txt")
        self.first_write = True

    def get_text_feature(self, img_name, cid, cls_name, prompt):

        prompt = prompt.lower()
        fg_names = [[ori_name] + syn_names for ori_name, syn_names in
                    zip(PascalVOCDataset.CLASSES, VOC_SYNONYMS)]
        tgt_names = list(PascalVOCDataset.CLASSES[1:]) + VOC_BACKGROUND_CATEGORY
        fg_bg_prompts = entity_replace(prompt, tgt_names, fg_names[cid], VOC_2GRAM[cid])
        if not fg_bg_prompts:
            # topk_writer.write_line(img_name, [0] * 5)
            write_mode = 'a'
            if self.first_write:
                self.first_write = False
                write_mode = 'w'
            with open(self.undeal_path, write_mode) as undeal_writer:
                undeal_writer.write(f"{img_name},{cls_name},{prompt}\n")
            return None
        text_tokens = clip.tokenize(fg_bg_prompts).cuda()
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features  # (#fg+#bg, D)


class TextFeatureFixed(TextFeatureBase):
    def __init__(self, model):
        super(TextFeatureFixed, self).__init__(model)
        fg_text_features = zeroshot_classifier(self.model, VOC_NEW, imagenet_templates)
        bg_text_features = zeroshot_classifier(self.model, VOC_BACKGROUND_CATEGORY, imagenet_templates)
        self.fg_bg_text_features = torch.cat((fg_text_features, bg_text_features), dim=0)

    def get_text_feature(self, *data_args, **data_kwargs):
        return self.fg_bg_text_features


def main():
    # nltk.download('punkt')
    os.makedirs(args.out, exist_ok=True)
    meta_path = osp.join(args.out, "meta.json")

    with open(meta_path, 'w') as fp:
        json.dump(vars(args), fp)

    num_fg_cls = len(VOC_NEW)
    num_bg_cls = len(VOC_BACKGROUND_CATEGORY)
    for i, c in enumerate(VOC_NEW + VOC_BACKGROUND_CATEGORY, 1):
        print(f"{i:03}: {c}")

    model, preprocess = clip.load("ViT-L/14@336px")
    model.cuda().eval()

    if args.text_method == 'ori':
        text_feature_getter = TextFeatureByOrigin(model, args.out)
    else:
        text_feature_getter = TextFeatureFixed(model)

    if args.img_method == 'whole':
        img_feature_getter = ImageFeatureWhole(model, preprocess)
    else:
        img_feature_getter = ImageFeatureBoxes(model, preprocess)

    with open(args.data_info, 'r') as fp:
        data_info = json.load(fp)

    cname2idx = {}
    for idx, name in enumerate(PascalVOCDataset.CLASSES):
        cname2idx[name] = idx

    out_path = osp.join(args.out, "clip_index.json")
    topk_writer = TopkClassIO(out_path)

    for di in tqdm(data_info):
        name = f"{di['img_index']:08}"
        cls_name = di['concept']
        cid = cname2idx[cls_name]

        img_path = osp.join(args.img, name + args.img_suffix)
        ann_path = osp.join(args.ann, name + args.ann_suffix) if args.ann is not None else None
        image_features = img_feature_getter.get_img_feature(img_path, ann_path)
        text_features = text_feature_getter.get_text_feature(name, cid, cls_name, di['prompt'])

        if text_features is None or image_features is None:
            topk_writer.write_line(name, [], [])
        else:
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            text_probs = text_probs.mean(dim=0).float()  # mean reduction on image dim
            top_probs, top_labels = text_probs.cpu().topk(args.topk, dim=-1)
            top_labels = top_labels.cpu().numpy()
            top_probs = top_probs.cpu().numpy()
            top_labels += 1
            # top_labels[top_labels > num_fg_cls] = 0
            topk_writer.write_line(name, top_labels.tolist(), top_probs.tolist())


if __name__ == '__main__':
    args = parse_args()
    main()
