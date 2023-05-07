import json
import os
import os.path as osp
import argparse
import json
import shutil
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/Zalick/Datasets/DiffuseMade')
    return parser.parse_args()


def main():
    dataset_list = [
        'fullData_raw_organized_Aphotoof_10k',
        'fullData_raw_organized_Aphotoof_shuffle_eta0.1_10k',
        'fullData_raw_organized_fullDescription_eta0.1_20k',
        'fullData_chatGPT_vary_10k',
        'fullData_chatGPT_new_massive_20k_005_075',
        'fullData_chatGPT_massive-v3_20k_005_075',
    ]

    output_dir = osp.join(args.root, "finalData")
    out_img_dir = osp.join(output_dir, "img_dir/train")
    out_ann_dir = osp.join(output_dir, "ann_dir/train")
    out_cross_dir = osp.join(output_dir, "cross_dir/train")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_ann_dir, exist_ok=True)
    os.makedirs(out_cross_dir, exist_ok=True)

    out_data_info_path = osp.join(output_dir, "data_infos.json")

    img_idx = 0
    out_data_info = []

    # check first
    for dataset_name in dataset_list:
        dataset_dir = osp.join(args.root, dataset_name)
        assert osp.isdir(dataset_dir), f"No such dir: {dataset_dir}"

    for dataset_name in dataset_list:
        dataset_dir = osp.join(args.root, dataset_name)
        img_dir = osp.join(dataset_dir, "img_dir/train")
        ann_dir = osp.join(dataset_dir, "ann_dir/train")
        cross_dir = osp.join(dataset_dir, "cross_dir/train")
        data_info_path = osp.join(dataset_dir, "data_infos.json")
        with open(data_info_path, 'r') as fp:
            data_info = json.load(fp)
        for di in tqdm(data_info, desc=dataset_name, total=len(data_info)):
            ori_name = f"{di['img_index']:08}"
            new_name = f"{img_idx:08}"
            di['img_index'] = img_idx
            img_idx += 1
            out_data_info.append(di)
            ori_img_path = osp.join(img_dir, ori_name + ".png")
            ori_ann_path = osp.join(ann_dir, ori_name + ".png")
            ori_cross_path = osp.join(cross_dir, ori_name + ".png")
            new_img_path = osp.join(out_img_dir, new_name + ".png")
            new_ann_path = osp.join(out_ann_dir, new_name + ".png")
            new_cross_path = osp.join(out_cross_dir, new_name + ".png")
            shutil.copyfile(ori_img_path, new_img_path)
            shutil.copyfile(ori_ann_path, new_ann_path)
            shutil.copyfile(ori_cross_path, new_cross_path)
        print(f"After {dataset_name}, img_idx={img_idx}")

    with open(out_data_info_path, 'w') as fp:
        json.dump(out_data_info, fp)

    print("Finished")


if __name__ == '__main__':
    args = parse_args()
    main()
