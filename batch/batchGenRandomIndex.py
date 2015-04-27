
"""
    generate train/dev/test idx.pkl for LJ2M dataset
"""

import os
import sys
sys.path.append('../')

import argparse
import pickle
import logging

from features.preprocessing import RandomIndex
from common import filename
from common import utils

def get_arguments(argv):

    parser = argparse.ArgumentParser(description='generate train/dev/test idx.pkl for LJ2M dataset')

    parser.add_argument('corpus_folder', metavar='corpus_folder', 
                        help='corpus folder which should be structured like LJ2M')  
    parser.add_argument('percent_train', metavar='percent_train', type=int,
                        help='percentage of training data')    
    parser.add_argument('percent_dev', metavar='percent_dev', type=int,
                        help='percentage of development data')   
    parser.add_argument('percent_test', metavar='percent_test', type=int,
                        help='percentage of testing data')  
    parser.add_argument('output_filename', metavar='output_filename', 
                        help='output file name')

    parser.add_argument('-v', '--verbose', action='store_true', default=False, 
                        help='show messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False, 
                        help='show debug messages')

    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':

    args = get_arguments(sys.argv[1:])

    if args.debug:
        loglevel = logging.DEBUG
    elif args.verbose:
        loglevel = logging.INFO
    else:
        loglevel = logging.ERROR
    logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s', level=loglevel) 
    logger = logging.getLogger(__name__)

    idx_dict = {}
    emotion_dirs = os.listdir(args.corpus_folder)

    for emotion in emotion_dirs:

        # we only process the 40 emotions appeared in LJ40K
        if emotion not in filename.emotions['LJ40K']:
            continue
        
        emotion_dir = os.path.join(args.corpus_folder, emotion)

        ndoc = len(os.listdir(emotion_dir)) - 2     # minus . and ..
        logger.info("emotion = %s, ndoc = %u", emotion, ndoc)

        set_dict = {}
        generator = RandomIndex(args.percent_train, args.percent_dev, args.percent_test)
        set_dict['train'], set_dict['dev'], set_dict['test'] = generator.shuffle(ndoc)

        idx_dict[emotion] = set_dict

        logger.info("ntrain = %u, ndev = %u, ntest = %u", len(idx_dict[emotion]['train']), len(idx_dict[emotion]['dev']), len(idx_dict[emotion]['test']))

        assert ndoc == len(idx_dict[emotion]['train']) + len(idx_dict[emotion]['dev']) + len(idx_dict[emotion]['test'])

    utils.save_pkl_file(idx_dict, args.output_filename)
