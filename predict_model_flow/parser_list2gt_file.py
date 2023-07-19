# Copyright (c) 2022 Innodisk Crop.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import argparse

def main():

    parser = argparse.ArgumentParser(description="gen gt_file from parser_image_list")
    parser.add_argument('-p', '--parser_list_txt', required=True, type=str, help='please give training data image txt include all image path')
    parser.add_argument('-c', '--classes_txt', required=True, type=str, help='please give training data image txt include all image path')
    args = parser.parse_args()

    with open(args.parser_list_txt, 'r') as f, open(args.classes_txt, 'r') as c, open('gt_list.txt', 'w') as g:
        _f = [_f.rstrip('\n') for _f in f]
        _c = [_c.rstrip('\n') for _c in c]
        print("laebl: {}".format(_c))
        for line in _f:
            name_start = line.rfind('/')
            name_end = line.rfind('.png')
            file_name = line[name_start+1:name_end]
            predict_start = line.find(' ')

            predict = line[predict_start+1:]
            _predict = predict.split(' ')

            for _p in range(len(_predict)):
                parse_predict = _predict[_p].replace(',', ' ')
                gt = file_name + ' ' + parse_predict
                _gt = gt[:-1] + _c[int(gt[-1])] + '\n'
                print(_gt)
                g.write(_gt)

if __name__ == "__main__":
    main()
