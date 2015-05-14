import os
import sys

import argparse
import logging

from shutil import copyfile

from common import utils
from common import filename
from common import output

def get_arguments(argv):

    parser = argparse.ArgumentParser(description='Training image models by using LJ2M')

    parser.add_argument('result_file', metavar='RESULT_FILE', 
                        help='results of batchTrainImageModels.py')
    parser.add_argument('model_folder', metavar='MODEL_FOLDER',     
                        help='the folder that contains input model files')
    parser.add_argument('scaler_folder', metavar='SCALER_FOLDER', 
                        help='the folder that contains input scaler files')
    
    parser.add_argument('-m', '--output_model_folder', metavar='OUTPUT_MODEL_FOLDER', default=output.get_folder_name_with_time('model40'), 
                        help='output folder for model files')
    parser.add_argument('-s', '--output_scaler_folder', metavar='OUTPUT_SCALER_FOLDER', default=output.get_folder_name_with_time('scaler40'), 
                        help='output folder for scaler files')

    # parser.add_argument('-e', '--emotion_ids', metavar='EMOTION_IDS', type=utils.parse_range, default=[0], 
    #                     help='a list that contains emotion ids ranged from 0-39 (DEFAULT: 0). This can be a range expression, e.g., 3-6,7,8,10-15')
    # parser.add_argument('-p', '--parameter_file', metavar='PARAMETER_FILE', default=None, 
    #                     help='a file include parameter C and gamma')
    # parser.add_argument('-k', '--svm_kernel', metavar='SVM_KERNEL', default='rbf', 
    #                     help='svm kernel type (DEFAULT: "rbf")')
    # parser.add_argument('-c', metavar='C', type=utils.parse_list, default=[1.0], 
    #                     help='SVM parameter (DEFAULT: 1). This can be a list expression, e.g., 0.1,1,10,100')
    # parser.add_argument('-g', '--gamma', metavar='GAMMA', type=utils.parse_list, default=[0.0003], 
    #                     help='RBF parameter (DEFAULT: 1/dimensions). This can be a list expression, e.g., 0.1,1,10,100')
    # parser.add_argument('-n', '--no_scaling', action='store_true', default=False,
    #                     help='do not perform feature scaling (DEFAULT: False)')
    # parser.add_argument('-r', '--no_predict', action='store_true', default=False,
    #                     help='do not perform prediction on dev data (DEFAULT: False)')

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

    results = output.PredictionResult(args.result_file)

    output.create_folder(args.output_model_folder)
    logger.info('output_model_folder = %s' % (args.output_model_folder))

    output.create_folder(args.output_scaler_folder)
    logger.info('output_scaler_folder = %s' % (args.output_scaler_folder))

    for e in filename.emotions['LJ40K']:

        c, gamma = results.get_params_by_emotion(e)
        model_fname = filename.get_model_filename(e, c, gamma)
        scaler_fname = filename.get_scaler_filename(e)

        # copy files
        src = os.path.join(args.model_folder, model_fname)
        dest = os.path.join(args.output_model_folder, model_fname)
        logger.info('copy file from "%s" to "%s"' % (src, dest))
        copyfile(src, dest)

        src = os.path.join(args.scaler_folder, scaler_fname)
        dest = os.path.join(args.output_scaler_folder, scaler_fname)
        logger.info('copy file from "%s" to "%s"' % (src, dest))
        copyfile(src, dest)

