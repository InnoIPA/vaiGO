'''
Copyright 2023 innodisk Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Author: 
  Hueiru, hueiru_chen@innodisk.com, innodisk Inc
  Jack, juihung_weng@innodisk.com, innodick Inc
'''


'''
Simple PyTorch LPRNet example - quantization
'''



import os
import sys
import argparse
import torch
from pytorch_nndct.apis import torch_quantizer

from pt_lpr_common import *
from pt_lpr_load_data import *


DIVIDER = '-----------------------------------------'


def quantize(quant_mode,batchsize,dset_dir,quant_model,weights_file):


  # use GPU if available   
  if (torch.cuda.device_count() > 0):
    print('You have',torch.cuda.device_count(),'CUDA devices available')
    for i in range(torch.cuda.device_count()):
      print(' Device',str(i),': ',torch.cuda.get_device_name(i))
    print('Selecting device 0..')
    device = torch.device('cuda:0')
  else:
    print('No CUDA devices available..selecting CPU')
    device = torch.device('cpu')

   # load trained model
  model = LPRNet().to(device)
  if (torch.cuda.device_count() > 0):
    model.load_state_dict(torch.load(weights_file))
  else:
    model.load_state_dict(torch.load(weights_file,map_location=torch.device('cpu')))

  # force to merge BN with CONV for better quantization accuracy
  optimize = 1

  # override batchsize if in test mode
  if quant_mode=='test':
    batchsize = 1

  rand_in = torch.randn([batchsize, 3, 24, 94])
  quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=quant_model) 
  quantized_model = quantizer.quant_model


  # data loader
  test_img_dirs = os.path.expanduser(dset_dir)
  test_loader = LPRDataLoader(test_img_dirs.split(','), [94,24], 7)
  # evaluate 
  test(quantized_model,device ,test_loader)


  # export config
  if quant_mode == 'calib':
    quantizer.export_quant_config()
  if quant_mode == 'test':
    quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)
  
  return



def run_main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-d', '--dset_dir',  type=str, help='Path to dataset path.')
  ap.add_argument('-qm', '--quant_model',  type=str, default='./',    help='output quantize model path')
  ap.add_argument('-w',  '--weights_file',  type=str, help='Path the weight_file.pt')
  ap.add_argument('-q',  '--quant_mode', type=str, default='calib',    choices=['calib','test'], help='Quantization mode (calib or test). Default is calib')
  ap.add_argument('-b',  '--batchsize',  type=int, default=1,        help='Testing batchsize - must be an integer. Default is 1')
  args = ap.parse_args()

  print('\n'+DIVIDER)
  print('PyTorch version : ',torch.__version__)
  print(sys.version)
  print(DIVIDER)
  print(' Command line options:')
  print ('--dset_dir       : ',args.dset_dir)
  print ('--quant_model    : ',args.quant_model)
  print ('--quant_mode     : ',args.quant_mode)
  print ('--weights        : ',args.weights_file)
  print ('--batchsize      : ',args.batchsize)
  print(DIVIDER)

  quantize(args.quant_mode,args.batchsize,args.dset_dir,args.quant_model,args.weights_file)

  return



if __name__ == '__main__':
    run_main()
