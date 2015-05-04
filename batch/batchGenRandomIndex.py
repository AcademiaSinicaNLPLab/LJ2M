
"""
    generate train/dev/test idx.pkl for LJ2M dataset
"""

import os
import sys

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

    parser.add_argument('-z', '--zero_vector_file', metavar='IGNORE_ZV', default=None, 
                        help='ignore the indexes by using zero-vector file; a pickle file made by Sven including indexs of zero vectors')
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

    # we only need 40 emotions appeared in LJ40K    
    emotion_dirs = filename.emotions['LJ40K']
    #emotion_dirs = os.listdir(args.corpus_folder)

    generator = RandomIndex(args.percent_train, args.percent_dev, args.percent_test, 
                            emotions=emotion_dirs, zero_vector_idxs_filename=args.zero_vector_file)
    idx_dict = {}
    idx_dict['train'], idx_dict['dev'], idx_dict['test'] = generator.shuffle(args.corpus_folder)

    logger.info("dumping file to %s" % (args.output_filename))
    utils.save_pkl_file(idx_dict, args.output_filename)
