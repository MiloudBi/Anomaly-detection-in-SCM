import numpy as np
import tensorflow as tf
import argparse
import pandas as pd
import os, sys
import timeit
import math
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

parser = argparse.ArgumentParser(description = 'MSCRED encoder-decoder')
parser.add_argument('--batch', type = int, default = 1,help = 'batch_size for data input') #current version supprts batch_size = 1
parser.add_argument('--sensor_n', type = int, default = 26, help = 'number of sensors')
parser.add_argument('--win_size', type = int, default = [10, 30, 60], help = 'window size of each segment')
parser.add_argument('--step_max', type = int, default = 5,   help = 'maximum step of ConvLSTM')
parser.add_argument('--learning_rate',  type=float, default= 0.0002, help='learning rate')
parser.add_argument('--training_iters',  type = int, default = 5,help = 'number of maximum training iterations')
parser.add_argument('--train_start_id',  type = int, default = 12, help = 'training start id')
parser.add_argument('--train_end_id',  type = int, default = 800, help = 'training end id')
parser.add_argument('--test_start_id',  type = int, default = 800, help = 'test start id')
parser.add_argument('--test_end_id',  type = int, default = 2000,help = 'test end id')
parser.add_argument('--save_model_step', type = int, default = 1,help = 'number of iterations to save model')
parser.add_argument('--model_path', type = str, default = '../models/',   help='path to save models')
parser.add_argument('--raw_data_path', type = str, default = '../data/ts_model_input.csv', help='path to load raw data')
parser.add_argument('--matrix_data_path', type = str, default = '../data/signature_matrix/', help='matrix data path')
# parser.add_argument('--test_input_path', type = str, default = '../data/matrix_data/test_data/',
# 				   help='test input data path')
parser.add_argument('--train_test_label', type=int, default = 1, help='train/test label: train (1), test (0)')
parser.add_argument('--GPU_id', type=int, default = 3,  help='GPU ID to select')
args = parser.parse_args()
print(args)