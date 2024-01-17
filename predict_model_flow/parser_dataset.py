import xml.etree.ElementTree as ET
from os import getcwd
import os

_classes = 'classes.txt'

with open(_classes, 'r') as f:
    classes = [_class for _class in f.read().splitlines()]

def convert_annotation(image_id, list_file):
    base_path, _ = os.path.splitext(image_id)
    in_file = open('{}.xml'.format(base_path))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
            
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()


image_ids = open('train.txt').read().strip().split()
list_file = open('parser_image_list.txt', 'w')
for image_id in image_ids:
    list_file.write(image_id)
    convert_annotation(image_id, list_file)
    list_file.write('\n')
list_file.close()
print('Gen parser_image_list.txt success!')
