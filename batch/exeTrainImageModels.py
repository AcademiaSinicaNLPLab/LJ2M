
import os
import sys

import numpy as np
import argparse
import pickle
import logging
import time

from sklearn.preprocessing import StandardScaler

from common import utils
from common import filename
from common import output
from features import preprocessing

def get_arguments(argv):

    parser = argparse.ArgumentParser(description='Training image models by using LJ2M')

    parser.add_argument('feature_list_file', metavar='FEATURE_LIST_FILE', 
                        help='This program will fuse the features listed in this file and feed all of them to the classifier. The file format is in JSON. See "feautre_list_ex.json" for example')
    parser.add_argument('index_file', metavar='INDEX_FILE', 
                        help='index file generated by batchGenRandomIndex.py')
    
    parser.add_argument('-o', '--output_folder', metavar='OUTPUT_FOLDER', default=None, 
                        help='output folder, if not specified, create it with name equal to system time')
    parser.add_argument('-e', '--emotion_ids', metavar='EMOTION_IDS', type=utils.parse_range, default=[0], 
                        help='a list that contains emotion ids ranged from 0-39 (DEFAULT: 0). This can be a range expression, e.g., 3-6,7,8,10-15')
    parser.add_argument('-p', '--parameter_file', metavar='PARAMETER_FILE', default=None, 
                        help='a file include parameter C and gamma')
    parser.add_argument('-k', '--svm_kernel', metavar='SVM_KERNEL', default='rbf', 
                        help='svm kernel type (DEFAULT: "rbf")')
    parser.add_argument('-c', metavar='C', type=utils.parse_list, default=[1.0], 
                        help='SVM parameter (DEFAULT: 1). This can be a list expression, e.g., 0.1,1,10,100')
    parser.add_argument('-g', '--gamma', metavar='GAMMA', type=utils.parse_list, default=[0.0003], 
                        help='RBF parameter (DEFAULT: 1/dimensions). This can be a list expression, e.g., 0.1,1,10,100')
    parser.add_argument('-n', '--no_scaling', action='store_true', default=False,
                        help='do not perform feature scaling (DEFAULT: False)')
    parser.add_argument('-r', '--no_predict', action='store_true', default=False,
                        help='do not perform prediction on dev data (DEFAULT: False)')

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
    if None != args.output_folder:
        if not os.path.exists(args.output_folder):
            logger.info('create output folder %s' % (args.output_folder))
            os.makedirs(args.output_folder)
    else:
        args.output_folder = output.create_folder_with_time('models')
        logger.info('create output folder %s' % (args.output_folder))

    # load features
    feature_list = preprocessing.FeatureList(args.feature_list_file)

    # load the index file
    idxs = utils.load_pkl_file(args.index_file)

    # create fused dataset
    fused_dataset = preprocessing.FusedDataset(idxs, loglevel=loglevel)

    for feature_name, data_path in feature_list:
        dataset = preprocessing.Dataset(data_path, loglevel=loglevel)
        fused_dataset.add_feature(feature_name, dataset)

    # read parameter file
    if args.parameter_file != None:
        param_dict = utils.read_parameter_file(args.parameter_file)

    # main loop
    best_res = {}

    if  'rbf' == args.svm_kernel:
        import model.svm as learner
        trainer = learner.SVM(loglevel=loglevel)
    elif 'linear' == args.svm_kernel:
        import model.linearsvm as learner
        trainer = learner.LinearSVM(loglevel=loglevel)
    else:
        raise ValueError('unsupported kernel')

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
            
            fpath = os.path.join(args.output_folder, filename.get_scaler_filename(emotion_name, 'pkl'))
            logger.info('dumpping scaler to %s' % (fpath))
            utils.save_pkl_file(scaler, fpath)

        best_res[emotion_name] = {}
        best_res[emotion_name]['score'] = 0

        if args.parameter_file != None:
            Cs = [param_dict[emotion_name][0]]
            gammas = [param_dict[emotion_name][1]]
        else:
            Cs = args.c
            gammas = args.gamma

        for c in Cs:
            for g in gammas:

                trainer.set(X=X_train, y=y_train, feature_name=fused_dataset.get_feature_name())
                temp_str = '[%s] start training: c=%f' % (emotion_name, c)
                if 'rbf' == args.svm_kernel:
                    temp_str += ', gamma=%f' % (g)
                logger.info(temp_str)

                start_time = time.time()
                trainer.train(C=c, kernel=args.svm_kernel, gamma=g, prob=True, random_state=np.random.RandomState(0))
                end_time = time.time()
                logger.info('[%s] training time = %f s' % (emotion_name, end_time-start_time))

                fpath = os.path.join(args.output_folder, filename.get_model_filename(emotion_name, c, g, 'pkl'))
                logger.info('[%s] dumpping model to %s' % (emotion_name, fpath))
                trainer.dump_model(fpath)

                if not args.no_predict:
                    result = trainer.predict(X_dev, y_dev, score=True, X_predict_prob=True, auc=True, decision_value=True)
                    if result['score'] > best_res[emotion_name]['score']:    
                        logger.info('save best result!!')                    
                        best_res[emotion_name]['gamma'] = g
                        best_res[emotion_name]['c'] = c
                        best_res[emotion_name]['results'] = result

    if not args.no_predict:
        fpath = os.path.join(args.output_folder, 'best_results_%s.pkl' % (str(args.emotion_ids)))
        logger.info('dumpping best results to %s' % (fpath))
        utils.save_pkl_file(best_res, fpath)           
        # ToDo: make csv file
