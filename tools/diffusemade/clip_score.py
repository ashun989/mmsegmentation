import os
import os.path as osp
import argparse
import torch
import clip
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
import nltk

from mmseg.utils import parse_path

from compare_labels import VOC_CLASSES

VOC_SYNONYMS = (
    [],
    ['aeroplanes', 'seaplane', 'lane'],
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


def one2two_gram(words):
    words2 = []
    for i in range(len(words) - 1):
        words2.append(' '.join([words[i], words[i + 1]]))
    return words2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True)
    parser.add_argument('--data-info', type=str, required=True)
    parser.add_argument('--img-suffix', type=str, default='.png')
    parser.add_argument('--out', type=str, required=True, help='dir to output file')
    # parser.add_argument('--split', type=str, default=None)
    return parser.parse_args()


def entity_replace(prompt, cid, cls_names, cls_syns, cls_two_grams):
    prompt_words = nltk.word_tokenize(prompt)  # one-gram
    cnames = [cls_names[cid]] + cls_syns[cid]
    replaced = []
    for cname in cnames:
        try:
            idx = prompt_words.index(cname)
            for i in range(1, len(cls_names)):  # skip background
                # if i != cid:
                prompt_words[idx] = cls_names[i]
                # else:
                #     prompt_words[idx] = cname
                replaced.append(' '.join(prompt_words))
            break
        except ValueError:
            pass
    if replaced:
        return replaced
    prompt_words2 = one2two_gram(prompt_words)
    cnames = cls_two_grams[cid]
    for cname in cnames:
        try:
            idx = prompt_words2.index(cname)
            for i in range(1, len(cls_names)):  # skip background
                prompt_words2[idx] = cls_names[i]
                text = prompt_words2[0]
                for word2 in prompt_words2[1:]:
                    text += word2.split(' ')[-1]
                replaced.append(text)
        except ValueError:
            pass
    return replaced


def get_text_probs(model, image_input, text_tokens, topk):
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    # similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    top_probs, top_labels = text_probs.cpu().topk(topk, dim=-1)
    top_labels += 1
    return top_labels


class TopkClassIO:
    def __init__(self, filename, write):
        self.filename = filename
        self.write = write
        self.fp = open(self.filename, 'w') if write else open(self.filename, 'r')

    def write_line(self, name, labels):
        self.fp.write(f"{name} ")
        for l in labels:
            self.fp.write(f"{l},")
        self.fp.write("\n")

    def read_all(self):
        results = {}
        while True:
            line = self.fp.readline()
            if not line:
                break
            name, label_str = line.strip().split()
            labels = list(map(int, label_str.split(',')[:-1]))
            results[name] = labels
        return results

    def close(self):
        self.fp.close()


def main():
    # nltk.download('punkt')

    model, preprocess = clip.load("ViT-L/14@336px")
    model.cuda().eval()

    # file_list = get_file_list(args.img, args.img_suffix, args.split)
    with open(args.data_info, 'r') as fp:
        data_info = json.load(fp)

    cname2idx = {}
    for idx, name in enumerate(VOC_CLASSES):
        cname2idx[name] = idx

    os.makedirs(args.out, exist_ok=True)
    out_path = osp.join(args.out, "clip_index.txt")
    undeal_path = osp.join(args.out, "undeal.txt")
    topk_writer = TopkClassIO(out_path, write=True)
    undeal_writer = open(undeal_path, 'w')

    for di in tqdm(data_info):
        name = f"{di['img_index']:08}"
        cls_name = di['concept']
        prompt = di['prompt'].lower()
        # if prompt[-1] == '.':
        #     prompt = prompt[:-1]
        # print(prompt)
        cid = cname2idx[cls_name]

        prompts = entity_replace(prompt, cid, VOC_CLASSES, VOC_SYNONYMS, VOC_2GRAM)
        if not prompts:
            # breakpoint()
            undeal_writer.write(f"{name},{cls_name},{prompt}\n")
            continue
        img_path = osp.join(args.img, name + args.img_suffix)
        img = Image.open(img_path).convert("RGB")
        img = preprocess(img)
        image_input = torch.tensor(np.stack([img])).cuda()
        text_tokens = clip.tokenize(prompts).cuda()
        top_labels = get_text_probs(model, image_input, text_tokens, 5)
        topk_writer.write_line(name, top_labels[0].cpu().numpy())

    topk_writer.close()
    undeal_writer.close()


if __name__ == '__main__':
    args = parse_args()
    main()
