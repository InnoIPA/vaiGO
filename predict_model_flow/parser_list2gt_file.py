# Copyright (c) 2022 Innodisk Crop.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate gt_file from parser_image_list")
    parser.add_argument('-p', '--parser_list_txt', required=True, type=str, help='Path to image list txt including all image paths')
    parser.add_argument('-c', '--classes_txt', required=True, type=str, help='Path to classes txt')
    args = parser.parse_args()

    with open(args.parser_list_txt, 'r') as f, open(args.classes_txt, 'r') as c, open('gt_list.txt', 'w') as g:
        _f = [line.strip() for line in f]
        _c = [line.strip() for line in c]
        print("laebl: {}".format(_c))
        for line in _f:
            parts = line.split(' ')
            file_path = parts[0]
            file_name = file_path[file_path.rfind('/')+1:file_path.rfind('.jpg')]
            
            for bbox_info in parts[1:]:
                bbox_parts = bbox_info.split(',')
                if len(bbox_parts) == 5:
                    # Extract bbox coordinates and class index
                    bbox = ' '.join(bbox_parts[:4])
                    class_index = int(bbox_parts[4])
                    class_label = _c[class_index] if class_index < len(_c) else 'unknown'

                    gt_line = f"{file_name} {bbox} {class_label}\n"
                    print(gt_line.strip())
                    g.write(gt_line)

if __name__ == "__main__":
    main()

