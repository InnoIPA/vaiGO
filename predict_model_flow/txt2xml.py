#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:MXY
# yolo的标签转换为labelImg形式的xml标签

import shutil
from xml.dom.minidom import Document
import os
import cv2
import argparse
import time

def class_dict(txt_path="classes.txt"):
    _dict = {}
    with open(txt_path, 'r') as f:
        _f = f.read().splitlines()
        for count, i in enumerate(_f):
            _dict[str(count)] = i
    return _dict

def get_image_files(folder_path):
    supported_formats = ['.jpg', '.jpeg', '.png']
    image_files = []
    for _format in supported_formats:
        image_files.extend([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(_format)])
    return image_files

def update_progress(current, total):
    percent = 100 * (current / total)
    if current >= total:
        print("\rFinished: {:.2f}%".format(100))
    else:
        print("\rFinished: {:.2f}%".format(percent), end="")
    time.sleep(0.01)


# def makexml(txtPath, xmlPath, picPath): 
def makexml(img_files, txtPath, xmlPath, class_dicts):
    total_images = len(img_files)
    percent = 0
    for name in img_files:
        xmlBuilder = Document()
        annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签
        xmlBuilder.appendChild(annotation)
        base_path, _formats = os.path.splitext(name)
        txtFile = open(base_path + ".txt")
        txtList = txtFile.readlines()
        img = cv2.imread(name)
        Pheight, Pwidth, Pdepth = img.shape
        for i in txtList:
            oneline = i.strip().split(" ")
            folder = xmlBuilder.createElement("folder")  # folder标签
            folderContent = xmlBuilder.createTextNode("VOC2007")
            folder.appendChild(folderContent)
            annotation.appendChild(folder)

            filename = xmlBuilder.createElement("filename")  # filename标签
            
            basename = os.path.basename(name)
            file_name = os.path.splitext(basename)[0]
            
            filenameContent = xmlBuilder.createTextNode(file_name+_formats)

            filename.appendChild(filenameContent)
            annotation.appendChild(filename)

            size = xmlBuilder.createElement("size")  # size标签
            width = xmlBuilder.createElement("width")  # size子标签width
            widthContent = xmlBuilder.createTextNode(str(Pwidth))
            width.appendChild(widthContent)
            size.appendChild(width)
            height = xmlBuilder.createElement("height")  # size子标签height
            heightContent = xmlBuilder.createTextNode(str(Pheight))
            height.appendChild(heightContent)
            size.appendChild(height)
            depth = xmlBuilder.createElement("depth")  # size子标签depth
            depthContent = xmlBuilder.createTextNode(str(Pdepth))
            depth.appendChild(depthContent)
            size.appendChild(depth)
            annotation.appendChild(size)

            object = xmlBuilder.createElement("object")
            picname = xmlBuilder.createElement("name")
            nameContent = xmlBuilder.createTextNode(class_dicts[oneline[0]])
            picname.appendChild(nameContent)
            object.appendChild(picname)
            pose = xmlBuilder.createElement("pose")
            poseContent = xmlBuilder.createTextNode("Unspecified")
            pose.appendChild(poseContent)
            object.appendChild(pose)
            truncated = xmlBuilder.createElement("truncated")
            truncatedContent = xmlBuilder.createTextNode("0")
            truncated.appendChild(truncatedContent)
            object.appendChild(truncated)
            difficult = xmlBuilder.createElement("difficult")
            difficultContent = xmlBuilder.createTextNode("0")
            difficult.appendChild(difficultContent)
            object.appendChild(difficult)
            bndbox = xmlBuilder.createElement("bndbox")
            xmin = xmlBuilder.createElement("xmin")
            mathData = int(((float(oneline[1])) * Pwidth + 1) - (float(oneline[3])) * 0.5 * Pwidth)
            xminContent = xmlBuilder.createTextNode(str(mathData))
            xmin.appendChild(xminContent)
            bndbox.appendChild(xmin)
            ymin = xmlBuilder.createElement("ymin")
            mathData = int(((float(oneline[2])) * Pheight + 1) - (float(oneline[4])) * 0.5 * Pheight)
            yminContent = xmlBuilder.createTextNode(str(mathData))
            ymin.appendChild(yminContent)
            bndbox.appendChild(ymin)
            xmax = xmlBuilder.createElement("xmax")
            mathData = int(((float(oneline[1])) * Pwidth + 1) + (float(oneline[3])) * 0.5 * Pwidth)
            xmaxContent = xmlBuilder.createTextNode(str(mathData))
            xmax.appendChild(xmaxContent)
            bndbox.appendChild(xmax)
            ymax = xmlBuilder.createElement("ymax")
            mathData = int(((float(oneline[2])) * Pheight + 1) + (float(oneline[4])) * 0.5 * Pheight)
            ymaxContent = xmlBuilder.createTextNode(str(mathData))
            ymax.appendChild(ymaxContent)
            bndbox.appendChild(ymax)
            object.appendChild(bndbox)

            annotation.appendChild(object)
    
        f = open(os.path.join(xmlPath, file_name + '.xml'), 'w')
        xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
        f.close()
        
        percent += 1 
        update_progress(percent, total_images)
        
            


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='test')
    parse.add_argument('-img', '--image_path', type=str)
    parse.add_argument('-txt', '--yolo_label_txt_path', type=str)
    parse.add_argument('-xml', '--output_xml_path', type=str)
    args = parse.parse_args()

    img_path = args.image_path
    txt_path = args.yolo_label_txt_path if args.yolo_label_txt_path else img_path
    xml_path = args.output_xml_path if args.output_xml_path else img_path
    class_dicts = class_dict()
    img_files = get_image_files(img_path)
    makexml(img_files, txt_path, xml_path, class_dicts)
    print('Gen xml success!')