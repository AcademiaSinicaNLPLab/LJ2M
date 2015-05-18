


import sys
import os
import argparse
import logging
import csv
import numpy as np
from collections import OrderedDict

from common import utils
from common import filename
from common import output
from features import preprocessing

from model import linearsvm
from model import svm

def get_arguments(argv):
    parser = argparse.ArgumentParser(description='predict document level')
    parser.add_argument('model_folder', metavar='MODEL_FOLDER', 
                        help='folder that contains models files')
    parser.add_argument('feature_list_file', metavar='FEATURE_LIST_FILE', 
                        help='This program will fuse the features listed in this file and feed all of them to the classifier. The file format is in JSON. See "feautre_list_ex.json" for example')

    parser.add_argument('-l', '--linear', action='store_true', default=False, 
                        help='use liblinear to predict')
    parser.add_argument('-e', '--emotion_ids', metavar='EMOTION_IDS', type=utils.parse_range, default=[0], 
                        help='a list that contains emotion ids ranged from 0-39 (DEFAULT: 0). This can be a range expression, e.g., 3-6,7,8,10-15')
    parser.add_argument('-s', '--scaler_folder', metavar='SCALER_FOLDER', default=None, 
                        help='folder that contains scaler files')
    parser.add_argument('-o', '--output_file', metavar='OUTPUT_FILE', default='output.csv', 
                        help='output csv file (DEFAULT: output.csv)')
    parser.add_argument('-i', '--index_file', metavar='INDEX_FILE', default=None, 
                        help='index file including testing set; if not specified, we test with all data')

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

    # pre-checking
    # if not os.path.exists(args.output_folder):
    #     logger.info('create output folder %s' % (args.output_folder))
    #     os.makedirs(args.output_folder)

    # load feature list
    feature_list = preprocessing.FeatureList(args.feature_list_file)
    emotions = filename.emotions['LJ40K']

    # create fused dataset
    idxs = utils.load_pkl_file(args.index_file) if args.index_file is not None else None
    fused_dataset = preprocessing.FusedDataset(idxs, loglevel=loglevel)

    for feature_name, data_path in feature_list:
        dataset = preprocessing.Dataset(data_path, loglevel=loglevel)
        fused_dataset.add_feature(feature_name, dataset)


    # load models
    if args.linear:
        from model import linearsvm
    else:
        from model import svm

    # main loop
    output_dict = OrderedDict()
    for emotion_id in args.emotion_ids:

        emotion_name = emotions[emotion_id]
        logger.info('predicting model for emotion "%s"' % emotion_name)

        # load learner
        learner = linearsvm.LinearSVM(loglevel=loglevel) if args.linear else svm.SVM(loglevel=loglevel)
        fpath = os.path.join(args.model_folder, filename.get_filename_by_emotion(emotion_name, args.model_folder))
        logger.info('loading model from %s' % (fpath))
        learner.load_model(fpath)

        # get data
        if args.index_file is None:
            # get all data
            X, y = fused_dataset.get_dataset(emotion_name, None)
        else:
            # get testing set
            X, y = fused_dataset.get_dataset(emotion_name, 'test')

        # load scaler
        if args.scaler_folder != None:
            fpath = os.path.join(args.scaler_folder, filename.get_filename_by_emotion(emotion_name, args.scaler_folder))
            logger.info('loading scaler from %s' % (fpath))
            scaler = utils.load_pkl_file(fpath)
            X = scaler.transform(X)

        # predicting
        logger.debug('predicting with "%s" classifier' % (emotion_name))
        results = learner.predict(X, y, score=True, X_predict_prob=True, auc=True, decision_value=True)
        output_dict[emotion_name] = (results['score'], results['auc'])


    # reorganize data
    output_list = []
    output_list.append(['']+output_dict.keys())
    output_list.append(['accuracy'] + [v[0] for k, v in output_dict.iteritems()])
    output_list.append(['auc'] + [v[1] for k, v in output_dict.iteritems()])
    
    # write output csv
    with open(args.output_file, 'w') as f:
        w = csv.writer(f)
        w.writerows(output_list)
                