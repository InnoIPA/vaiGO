#!/usr/bin/python3
# Copyright (c) 2023 innodisk Crop.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import random
import argparse
import glob
from pathlib import Path

t_image_path = 'train.txt'
v_image_path = 'val.txt'
error_msg_ratio = 'Wrong ratio {} ratio need to be a float in 0.1 ~ 0.9'
error_msg_path = 'Invalid image_path {}'

def clear_train_val():
    if Path(t_image_path).is_file():
        os.remove(t_image_path)
    if Path(v_image_path).is_file():
        os.remove(v_image_path)

def parser():
    parser = argparse.ArgumentParser(description="split data to train.txt val.txt")
    parser.add_argument('image_dir_path', nargs='+', help='Please only give current directory Usage: splitdata.py <image directory for training> ...')
    parser.add_argument('-r', "--ratio", type=float, default=0.9,
                        help="please give the ratio to split e.g. 0.9 will have 90%% image in train.txt 10%% in val.txt")
    return parser

def check_arguments_errors(args):
    for path in args.image_dir_path:
        if not os.path.exists(path):
            raise ValueError(error_msg_path.format(path))
    if not (round(args.ratio, 1) <= 1 and round(args.ratio, 1) >= 0):
        raise ValueError(error_msg_ratio.format(args.ratio))

def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        raise ValueError('Please give the image directory')
    elif input_path_extension == "txt":
        raise ValueError('Please give the image directory')
    else:
        return glob.glob(os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))

def check_image_exist(image_list):
    if image_list:
        pass
    else:
        raise ValueError('Please give the image directory which have .png .jpg or .jpeg inside.')

def splitdata(image_path_list, ratio=0.9):
    for image_path in image_path_list:
        files = load_images(os.path.abspath(image_path))
        check_image_exist(files)
        random.shuffle(files)
        cut = int(len(files)*round(ratio, 1))
        arr1 = files[:cut]
        arr2 = files[cut:]

        with open(t_image_path, 'a+') as train, open(v_image_path, 'a+') as val:
            train.writelines('\n'.join(s for s in arr1))
            val.writelines('\n'.join(s for s in arr2))

def check_splitdata_success():
    with open(t_image_path) as t, open(v_image_path) as v:
        train_image = t.read().splitlines()
        val_image = v.read().splitlines()
        # val_image = v.readlines() #FAIL situation it will have '\n'
    for _t in train_image:
        if not os.path.isfile(_t):
            print(f'Fail: {t_image_path} {_t} image not found')
            return 0
    for _v in val_image:
        if not os.path.isfile(_v):
            print(f'Fail: {v_image_path} {_v} image not found')
            return 0
    print('Splitdata success!')

def main():
    args = parser().parse_args()
    clear_train_val()
    check_arguments_errors(args)
    splitdata(args.image_dir_path, args.ratio)
    check_splitdata_success()

if __name__ == '__main__':
    main()