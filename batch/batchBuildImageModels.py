
import os
import sys
sys.path.append('../')

import numpy as np
import argparse
import pickle
import logging
import time

from sklearn.preprocessing import StandardScaler

from common import utils
from common import filename
from features import preprocessing
from model import learner

def get_arguments(argv):

    parser = argparse.ArgumentParser(description='Training image models by using LJ2M')

    parser.add_argument('feature_list_file', metavar='FEATURE_LIST_FILE', 
                        help='This program will fuse the features listed in this file and feed all of them to the classifier. The file format is in JSON. See "feautre_list_ex.json" for example')
    parser.add_argument('index_file', metavar='INDEX_FILE', 
                        help='index file generated by batchGenRandomIndex.py')
    parser.add_argument('output_folder', metavar='OUTPUT_FOLDER', 
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


    # load features
    feature_list = preprocessing.FeatureList(args.feature_list_file)

    # load the index file
    idxs = utils.load_pkl_file(args.index_file)

    # create fused dataset
    fused_dataset = preprocessing.FusedDataset(idxs, loglevel=loglevel)

    for feature_name, data_path in feature_list:
        dataset = preprocessing.Dataset(data_path, loglevel=loglevel)
        fused_dataset.add_feature(feature_name, dataset)


    # main loop
    best_res = {}

    for emotion_id in args.emotion_ids:

        emotion_name = filename.emotions['LJ40K'][emotion_id]
        logger.info('training model for emotion "%s"' % emotion_name)

        X_train, y_train = fused_dataset.get_dataset(emotion_name, 'train')
        X_dev, y_dev = fused_dataset.get_dataset(emotion_name, 'dev')

        if not args.no_scaling:
            scaler = StandardScaler(with_mean=True, with_std=True)
            logger.debug("applying standard scaling")
            X_train = scaler.fit_transform(X_train)
            X_dev = scaler.transform(X_dev)
            
            fpath = os.path.join(args.output_folder, 'scaler_%s.pkl' % (emotion_name))
            logger.info('dumpping scaler to %s' % (fpath))
            utils.save_pkl_file(scaler, fpath)

        best_res[emotion_name] = {}
        best_res[emotion_name]['score'] = 0
        for c in args.c:
            for g in args.gamma:

                # we do not do scaling in learner
                l = learner.SVM(X=X_train, y=y_train, feature_name=fused_dataset.get_feature_name(), scaling=False, loglevel=loglevel)

                logger.info('[%s] start training: c=%f, gamma=%f' % (emotion_name, c, g))
                start_time = time.time()
                l.train(C=c, kernel='rbf', gamma=g, prob=True, random_state=np.random.RandomState(0))
                end_time = time.time()
                logger.info('[%s] training time = %f s' % (emotion_name, end_time-start_time))

                fpath = os.path.join(args.output_folder, 'model_%s_c%f_g%f.pkl' % (emotion_name, c, g))
                logger.info('[%s] dumpping model to %s' % (emotion_name, fpath))
                l.dump_model(fpath)

                result = l.predict(X_dev, y_dev, score=True, X_predict_prob=True, auc=True)
                if result['score'] > best_res[emotion_name]['score']:
                    best_res[emotion_name]['score'] = result['score']
                    best_res[emotion_name]['gamma'] = g
                    best_res[emotion_name]['c'] = c
                    best_res[emotion_name]['X_predict_prob'] = result['X_predict_prob']
                    best_res[emotion_name]['auc'] = result['auc']

        logger.info("[%s] best score = %f" % (emotion_name, best_res[emotion_name]['score']))
        logger.info("[%s] best gamma = %f" % (emotion_name, best_res[emotion_name]['gamma']))
        logger.info("[%s] best c = %f" % (emotion_name, best_res[emotion_name]['c']))
        logger.info("[%s] best prob = %f" % (emotion_name, best_res[emotion_name]['X_predict_prob']))
        logger.info("[%s] best auc = %f" % (emotion_name, best_res[emotion_name]['auc']))



    fpath = os.path.join(args.output_folder, 'best_results.pkl')
    logger.info('dumpping best results to %s' % (fpath))
    utils.save_pkl_file(best_res, fpath)           
    # ToDo: make csv file
