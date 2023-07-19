import os
import argparse
import time
import numpy as np
import logging
import datetime

from mod.predictor import PREDICTOR
from mod.util import open_json

date = datetime.date.today()

CFG     = '../config.json'
DISPLAY = "DISPLAY"
WIDTH   = "WIDTH"
HEIGHT  = "HEIGHT"
divider = '------------------------------------'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path_txt' , type=str, required=True, help='Path to a image which you would like to inference.')
    parser.add_argument('-x', '--xmodel', type=str, required=True, default='yolo', help='Type of xmodel, default is yolo')
    parser.add_argument('-t', '--target', type=str, help='Output type, default is dp')
    args = parser.parse_args()


    args = parser.parse_args()
    cfg = open_json(CFG)
    pred = PREDICTOR(args, cfg)
    
    logging.basicConfig(level=logging.INFO)

    logging.info(divider)
    logging.info(" Command line options:")
    logging.info(" Date:    {}".format(date))
    

    ''' Select Input '''
    if args.image_path_txt:
        if os.path.isfile(args.image_path_txt):
            logging.info(' --input     : {}'.format('image_path_txt'))
            pred.get_frame = pred.path_txt_image_get

        else:
            print("Could not find the file {}.".format(args.image))
            return

    ''' Select Xmodel '''
    if args.xmodel == 'cnn':
        logging.info(' --model     : {}'.format('cnn'))
        pred.init_model = pred.init_cnn
        pred.run_model = pred.run_cnn

    else:
        logging.info(' --model     : {}'.format('yolo'))
        pred.init_model = pred.init_yolo
        pred.run_model = pred.run_yolo
    
    ''' Select Output '''
    if args.target == 'dp':
        if not os.path.exists(DISPLAY_CARD_PATH):
            logging.info('Error: zynqmp-display device is not ready.')
            return

        ''' Resolution check '''
        width = int(pred.cfg[DISPLAY][WIDTH])
        height = int(pred.cfg[DISPLAY][HEIGHT])
        
        resolution = "{}x{}".format(width, height)
        all_res = os.popen("modetest -M xlnx -c| awk '/name refresh/ {f=1;next}  /props:/{f=0;} f{print $1 \"@\" $2}'").read()
        
        if all_res.find(resolution) == -1:
            print("\nError: Monitor doesn't support resolution {}".format(resolution))
            print("All supported resolution:\n{}".format(all_res))
            return

        logging.info(' --output    : {}'.format('dp'))
        pred.output = pred.dp_out

    elif args.target == 'image':
        logging.info(' --output    : {}'.format('image'))
        pred.output = pred.image_out

    elif args.target == 'video':
        logging.info(' --output    : {}'.format('video'))
        pred.output = pred.video_out
    
    elif args.target == 'result_txt':
        logging.info(' --output    : {}'.format('result_txt'))
        pred.output = pred.result_txt_out

    pred.predict()
if __name__ == '__main__':
    main()