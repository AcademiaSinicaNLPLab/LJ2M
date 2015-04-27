
import os
import sys
sys.path.append('../')

import numpy as np
import argparse
import pickle
import logging

from common import utils
from common import filename
from features import preprocessing

def get_arguments(argv):

    parser = argparse.ArgumentParser(description='Training image models by using LJ2M')

    parser.add_argument('feature_list_file', metavar='feature_list_file', 
                        help='This program will fuse the features listed in this file and feed all of them to the classifier. The file format is in JSON. See "feautre_list_ex.json" for example')
    parser.add_argument('output_folder', metavar='output_folder', 
                        help='the folder to store model files')
    
    parser.add_argument('-e', '--emotion_ids', metavar='EMOTION_IDS', type=utils.parse_range, default=[0], 
                        help='a list that contains emotion ids ranged from 0-39 (DEFAULT: 0). This can be a range expression, e.g., 3-6,7,8,10-15')
    parser.add_argument('-c', metavar='C', type=utils.parse_list, default=[1.0], 
                        help='SVM parameter (DEFAULT: 1). This can be a list expression, e.g., 0.1,1,10,100')
    parser.add_argument('-g', '--gamma', metavar='GAMMA', type=utils.parse_list, default=None, 
                        help='RBF parameter (DEFAULT: 1/dimensions). This can be a list expression, e.g., 0.1,1,10,100')
    parser.add_argument('-n', '--no_scaling', action='store_true', default=False,
                        help='do not perform feature scaling (DEFAULT: False)')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, 
                        help='show messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False, 
                        help='show debug messages')

    args = parser.parse_args(argv)
    return args

    
if __name__ == '__main__':
    
    args = get_arguments(sys.argv[1:])
    features = preprocessing.get_feature_list(args.feature_list_file)

    if args.debug:
        loglevel = logging.DEBUG
    elif args.verbose:
        loglevel = logging.INFO
    else:
        loglevel = logging.ERROR
    logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s', level=loglevel) 
    logger = logging.getLogger(__name__)

    # some pre-checking
    if args.output_folder is not None and not os.path.isdir(args.output_folder):
        raise Exception("output folder %s doesn't exist." % (args.output_folder))


    for emotion_id in args.emotion_ids:

        emotion_name = filename.emotions.LJ40K[emotion_id]