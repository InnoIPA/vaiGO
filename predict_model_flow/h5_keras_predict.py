from h5_keras_predict_yolo import YOLO
from PIL import Image
import tensorflow as tf
import time
import os
from tqdm import tqdm
import cv2
import argparse

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    

def write_items_to_file(image_id, items, fw):
    for item in items:
        fw.write(image_id + " " + " ".join([str(comp) for comp in item]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="split data to train.txt val.txt")
    parser.add_argument('-t', "--train_txt", type=str, required=True, help='please give training data image txt include all image path')
    parser.add_argument('-m', "--h5_model", type=str, required=True, help='please give your model.h5')
    parser.add_argument('-s', "--size", type=int, required=True, help='please give input image size')
    parser.add_argument('-a', "--anchors", type=str, required=True, help='please give training anchor txt file')
    parser.add_argument('-c', "--classes", type=str, required=True, help='please give classes txt file')
    parser.add_argument('-o', "--output_result", type=str, required=True, help='please give output folder')
    args = parser.parse_args()

    info = {
        "model_path"        : args.h5_model,
        "anchors_path"      : args.anchors,
        "classes_path"      : args.classes,
        "score"             : 0.5,
        "iou"               : 0.45,
        "eager"             : False,
        "max_boxes"         : 100,
        # 显存比较小可以使用416x416
        # 显存比较大可以使用608x608
        "model_image_size"  : (args.size, args.size),
        "output_path"       : args.output_result
    }

    h5_yolo = YOLO(info)
    if not os.path.exists(args.output_result):
        os.makedirs(args.output_result)
    with open(args.train_txt) as fr:
        lines = fr.readlines()
    fw = open('h5_result.txt', "w")
    for line in tqdm(lines):
        img_path = line.strip().split(" ")[0]
        fname = os.path.split(img_path)[-1]
        image_id = os.path.splitext(fname)[0]
        image = Image.open(img_path)
        # image = cv2.imread(img_path)
        items = h5_yolo.detect_image(image)
        print(items)
        write_items_to_file(image_id, items, fw)
    fw.close()
