import sys
sys.path.append( "../" )

import os
import argparse
import logging
import csv

from common import utils
from common import filename
from common import output
from model.learner import SVM

def get_arguments(argv):
    parser = argparse.ArgumentParser(description='predict sentence probabilities')
    parser.add_argument('model_folder', metavar='MODEL_FOLDER', 
                        help='folder that contains models files')
    parser.add_argument('feature_folder', metavar='FEATURE_FOLDER', 
                        help='folder that contains feature files')

    parser.add_argument('-e', '--emotion_ids', metavar='EMOTION_IDS', type=utils.parse_range, default=[0], 
                        help='a list that contains emotion ids ranged from 0-39 (DEFAULT: 0). This can be a range expression, e.g., 3-6,7,8,10-15')
    parser.add_argument('-s', '--scaler_folder', metavar='SCALER_FOLDER', default=None, 
                        help='folder that contains scaler files')
    parser.add_argument('-o', '--output_folder', metavar='OUTPUT_FOLDER', default=None, 
                        help='output folder, if not specified, create it with name equal to system time')

    parser.add_argument('-v', '--verbose', action='store_true', default=False, 
                        help='show messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False, 
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args

def output_emotions(filename, probs, emotions):

    n_sentence = len(probs[probs.keys()[0]])
    output_list = []

    # do transpose
    for i in range(n_sentence):
        sentence_list = []
        for emotion in emotions:
            sentence_list.append(probs[emotion][i])
        output_list.append(sentence_list)

    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(output_list)


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
    if None != args.output_folder and not os.path.exists(args.output_folder):
        logger.info('create output folder %s' % (args.output_folder))
        os.makedirs(args.output_folder)
    else:
        args.output_folder = output.create_folder_with_time('output')
        logger.info('create output folder %s' % (args.output_folder))

    emotions = filename.emotions['LJ40K']


    # load models
    learners = {}
    for emotion in emotions:
        learners[emotion] = SVM()
        
        if args.scaler_folder != None:
            fpath = os.path.join(args.scaler_folder, filename.get_filename_by_emotion(emotion, args.scaler_folder))
            learners[emotion].load_scaler(fpath)

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
        fpath = os.path.join(args.feature_folder, filename.get_filename_by_emotion(emotion, args.feature_folder))
        logger.info('loading test data %s' % (fpath))
        test_data = utils.load_pkl_file(fpath)

        
        for doc_idx in range(len(test_data)):
            logger.debug('predicting doc %u' % (doc_idx))

            # init result matrix
            probs = {}

            # predict on 40 models
            for emotion in emotions:
                logger.debug('predicting with "%s" classifier' % (emotion))
                results = learners[emotion].predict(test_data[doc_idx]['X'], None, X_predict_prob=True)
                probs[emotion] = results['X_predict_prob'].tolist()

            # output csv
            fpath = os.path.join(emotion_dir, '%u.csv' % (doc_idx))
            logger.debug('writing probs to %s' % (fpath))
            # we output the file in LJ40K_feelingwheel order
            output_emotions(fpath, probs, filename.emotions['LJ40K_feelingwheel'])
                