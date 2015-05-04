
import sys
import os
import argparse
import logging

from common import utils
from common import filename
from common import output

def get_arguments(argv):
    parser = argparse.ArgumentParser(description='convet data from csv to png')
    parser.add_argument('input_folder', metavar='INPUT_FOLDER', 
                        help='folder that contains prob. csv files')

    # parser.add_argument('-e', '--emotion_ids', metavar='EMOTION_IDS', type=utils.parse_range, default=[0], 
    #                     help='a list that contains emotion ids ranged from 0-39 (DEFAULT: 0). This can be a range expression, e.g., 3-6,7,8,10-15')
    parser.add_argument('-o', '--output_folder', metavar='OUTPUT_FOLDER', default=None, 
                        help='output folder, if not specified, create it with name equal to system time')

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
    logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s') 
    logger = logging.getLogger(__name__)
    logger.setLevel(loglevel)

    # pre-checking
    if None != args.output_folder and not os.path.exists(args.output_folder):
        logger.info('create output folder %s' % (args.output_folder))
        os.makedirs(args.output_folder)
    else:
        args.output_folder = output.create_folder_with_time('image')
        logger.info('create output folder %s' % (args.output_folder))

    # input
    emotion_dirs = os.listdir(args.input_folder)
    for emotion in emotion_dirs:

        emotion_dir = os.path.join(args.input_folder, emotion)
        csvfiles = os.listdir(emotion_dir)

        for csvfile in csvfiles:
            emotion_prob = output.EmotionProb(emotions=filename.emotions['LJ40K_feelingwheel'], loglevel=loglevel)
            fpath = os.path.join(emotion_dir, csvfile)
            emotion_prob.load_csv(fpath)

            # output
            fn, ext = os.path.splitext(csvfile)
            emotion_output_dir = fpath = os.path.join(args.output_folder, emotion)
            if not os.path.exists(emotion_output_dir):
                os.makedirs(emotion_output_dir)
            fpath = os.path.join(emotion_output_dir, fn+'.png')

            if os.path.exists(fpath):
                raise ValueError('file %s existed ' % (fpath))

            # ToDo: fine tune prob_threshold
            emotion_prob.dump_png(fpath, color_background=(255, 255, 255), alpha=True, prob_theshold=0.5)

