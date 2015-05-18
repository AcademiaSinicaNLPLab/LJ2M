import sys
import os
import argparse
import logging
import csv
import numpy as np

from common import utils
from common import filename
from common import output
from features import preprocessing

from model import linearsvm
from model import svm

def get_arguments(argv):
    parser = argparse.ArgumentParser(description='predict sentence probabilities')
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
    parser.add_argument('-o', '--output_folder', metavar='OUTPUT_FOLDER', default=output.get_folder_name_with_time('proba'), 
                        help='output folder, if not specified, create it with name equal to system time')
    parser.add_argument('-i', '--dump_png', action='store_true', default=False, 
                        help='dump .png images (Default: False)')
    parser.add_argument('-p', '--prob_threshold', metavar='PROB_THRESHOLD', default=0.5, 
                        help='threshold to filter image colors (Default: 0.5).')

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
    if not os.path.exists(args.output_folder):
        logger.info('create output folder %s' % (args.output_folder))
        os.makedirs(args.output_folder)

    # load feature list
    feature_list = preprocessing.FeatureList(args.feature_list_file)
    emotions = filename.emotions['LJ40K']

    # load models
    if args.linear:
        from model import linearsvm
    else:
        from model import svm

    learners = {}
    scalers = {}
    for emotion in emotions:
        learners[emotion] = linearsvm.LinearSVM(loglevel=loglevel) if args.linear else svm.SVM(loglevel=loglevel)
        
        if args.scaler_folder != None:
            fpath = os.path.join(args.scaler_folder, filename.get_filename_by_emotion(emotion, args.scaler_folder))
            scalers[emotion] = utils.load_pkl_file(fpath)

        fpath = os.path.join(args.model_folder, filename.get_filename_by_emotion(emotion, args.model_folder))
        logger.info('loading model for emotion %s' % (emotion))
        learners[emotion].load_model(fpath)


    # main loop
    for emotion_id in args.emotion_ids:

        emotion_name = emotions[emotion_id]
        logger.info('predicting model for emotion "%s"' % emotion_name)

        # create output dir
        emotion_dir =  os.path.join(args.output_folder, emotion_name)
        logger.info('create output folder %s' % (emotion_dir))
        if not os.path.exists(emotion_dir):
            os.makedirs(emotion_dir)

        # load test data
        feature_files = [(feature_name, preprocessing.FeatureList.get_full_data_path(emotion_name, data_path)) for feature_name, data_path in feature_list]
        fused_features = preprocessing.FusedDocSentence(feature_files, loglevel=loglevel)
        
        for doc_idx in range(fused_features.get_num_doc()):
            logger.debug('predicting doc %u' % (doc_idx))

            X_test = fused_features.get_fused_feature_vector_by_idx(doc_idx)
            if args.scaler_folder != None:
                X_test = scalers[emotion_name].transform(X_test)

            # init result matrix
            probs = []
            n_sentence = X_test.shape[0]
            y_test = np.array([1]*n_sentence)
            for i in range(n_sentence):
                probs.append(dict.fromkeys(emotions))

            # predict on 40 models
            for classifier_emotion in emotions:
                logger.debug('predicting with "%s" classifier' % (classifier_emotion))
                results = learners[classifier_emotion].predict(X_test, y_test, X_predict_prob=True)
                prob_list = results['X_predict_prob']
                for i in range(n_sentence):
                    probs[i][classifier_emotion] = prob_list[i] 

            # output csv
            # we output the file in LJ40K_feelingwheel order
            fpath = os.path.join(emotion_dir, '%u.csv' % (doc_idx))
            emotion_prob = output.EmotionProb(emotions=filename.emotions['LJ40K_feelingwheel'], probs=probs, loglevel=loglevel)
            emotion_prob.dump_csv(fpath)

            if args.dump_png:
                fpath = os.path.join(emotion_dir, '%u.png' % (doc_idx))
                emotion_prob.dump_png(fpath, color_background=(255, 255, 255), alpha=True, prob_theshold=args.prob_threshold)
                