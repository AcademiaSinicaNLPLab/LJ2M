
import os
import csv
import time
import logging
import pymongo

from PIL import Image, ImageDraw

import common.filename as naming
from common import utils


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def get_folder_name_with_time(prefix):
    t = time.strftime('%y%m%d%H%M%S', time.localtime(time.time()))
    name = '%s_%s' % (prefix, t)
    return name

def create_folder_with_time(prefix):
    name = get_folder_name_with_time(prefix)

    if not os.path.exists(name):
        os.makedirs(name)
    else:
        raise ValueError("folder %s exist" % (name))
    return name

def dump_dict_to_csv(file_name, data):
    w = csv.writer(open(file_name, 'w'))
    for key, val in data.items():
        w.writerow([key, val])

def dump_list_to_csv(file_name, data):
    w = csv.writer(open(file_name, 'w'))
    for row in data:
        w.writerow(row)


class PredictionResult:
    """
    self.results[emotion_name]['gamma']
    self.results[emotion_name]['c']
    self.results[emotion_name]['results']['score']
    self.results[emotion_name]['results']['auc']
    """
    def __init__(self, result_files=[], **kwargs):

        if type(result_files) is list:
            self.load_results(result_files)
        elif type(result_files) is str:
            self.load_results([result_files])
        else:
            raise ValueError('result_files type error' )

    def load_results(self, result_files=[]):
        self.results = {}
        for f in result_files:
            res = utils.load_pkl_file(f)
            self.results.update(res)

    def dump_results(self, filename):
        utils.save_pkl_file(self.results, filename)

    def _write_csv(self, filename, data):
        with open(filename, 'w') as f:
            w = csv.writer(f)
            w.writerows(data)

    def dump_accuracy(self, filename):
        data = []
        data.append(naming.emotions['LJ40K'])
        data.append([self.results[e]['results']['score'] for e in naming.emotions['LJ40K']])
        self._write_csv(filename, data)
        
    def dump_auc(self, filename):
        data = []
        data.append(naming.emotions['LJ40K'])
        data.append([self.results[e]['results']['auc'] for e in naming.emotions['LJ40K']])
        self._write_csv(filename, data)

    def dump_summary(self, filename):
        data = []
        data.append(['', 'AVERAGE']+naming.emotions['LJ40K'])

        scores = [self.results[e]['results']['score'] for e in naming.emotions['LJ40K']]
        average_score = float(sum(scores))/len(scores)
        data.append(['ACCURACY', average_score] + scores)

        aucs = [self.results[e]['results']['auc'] for e in naming.emotions['LJ40K']]
        average_auc = float(sum(aucs))/len(aucs)
        data.append(['AUC', average_auc] + aucs)

        Cs = [self.results[e]['c'] for e in naming.emotions['LJ40K']]
        data.append(['C', ''] + Cs)

        gammas = [self.results[e]['gamma'] for e in naming.emotions['LJ40K']]
        data.append(['gamma', ''] + gammas)

        self._write_csv(filename, data)

    def get_params_by_emotion(self, emotion):
        return (float(self.results[emotion]['c']), float(self.results[emotion]['gamma']))


class EmotionProb:
    """
    emotion probability input/output file

    options:
        loglevel
        mongo_addr

    """
    def __init__(self, emotions=None, probs=[], **kwargs):
        """
            probs: list of dictionary with keys=emotions
            emotions: should be ordered string list
        """
        loglevel = logging.ERROR if 'loglevel' not in kwargs else kwargs['loglevel']
        logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s')
        self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__) 
        self.logger.setLevel(loglevel)

        self.emotions = emotions
        self.probs = probs
        self.color_map = None   

    def dump_csv(self, filename):
        n_sentence = len(self.probs)
        output_list = []

        # do transpose
        for i in range(n_sentence):

            # to make sure the order, don't use self.probs[i].values()
            temp_row = [self.probs[i][emotion] for emotion in self.emotions]
            output_list.append(temp_row)

        with open(filename, 'w') as f:
            self.logger.info('writing to %s' % (filename))
            writer = csv.writer(f)
            writer.writerows(output_list)

    def load_csv(self, filename):

        assert self.emotions != None
        #for emotion in self.emotions:
        #    self.probs[emotion] = []

        with open(filename, 'r') as f:
            self.logger.info('loading from %s' % (filename))
            probs = []
            reader = csv.reader(f)
            for row in reader:
                prob = {e: float(p) for e, p in zip(self.emotions, row)}
                probs.append(prob)
        self.probs = probs

    def dump_png(self, filename, color_background=(255, 255, 255), alpha=True, prob_theshold=0.5):
        
        if self.color_map is None:
            self._get_colorinfo_from_db()

        assert self.probs != None

        color_matrix = []
        for prob_dict in self.probs:

            # background color
            render_colors = {e: color_background for e in self.emotions}

            # filter by probability
            filtered_prob_dict = {e: prob_dict[e] for e in self.emotions if prob_dict[e] > prob_theshold}

            # update render_colors
            for e in filtered_prob_dict:
                if alpha:
                    alpha_value = filtered_prob_dict[e]
                    rgba = tuple( list(self.color_map[e]['rgb']) + [alpha_value] )
                    rgb = utils.rgba_to_rgb(rgba, bg=color_background)
                else:
                    assert True
                    rgb = tuple(self.color_map[e]['rgb'])            
                    
                render_colors[e] = rgb

            order_render_colors = [(e, render_colors[e]) for e in self.emotions]
            
            row = []

            for emotion, color in order_render_colors:
                row.append(color)
            color_matrix.append(row)

        image = self._draw(color_matrix)

        ## save image
        self.logger.info('save image %s' % (filename))
        image.save(filename)

    #def load_png(self, filename):
    #    pass

    def _draw(self, color_matrix, cell_width=1, cell_height=1):
        # draw image
        image_width = cell_width*len(color_matrix[0])
        image_height = cell_height*len(color_matrix)

        image = Image.new('RGB', (image_width, image_height))

        ## draw a rectangle
        draw = ImageDraw.Draw(image)

        for row in range(len(color_matrix)):
            for col in range(len(color_matrix[row])):

                x0, x1 = col*cell_height, (col+1)*cell_height
                y0, y1 = row*cell_width, (row+1)*cell_width                
                draw.rectangle(xy=[(x0,y0),(x1,y1)], fill=color_matrix[row][col])
        return image

    def _get_colorinfo_from_db(self, 
                                mongo_addr='doraemon.iis.sinica.edu.tw', 
                                color_order=('feelit', 'color.order'), 
                                color_map=('feelit', 'color.map'), 
                                color_theme='black'):

        ### connect to mongodb
        mongo_conn = pymongo.MongoClient(mongo_addr)

        collection_color_order = mongo_conn[color_order[0]][color_order[1]]
        collection_color_map = mongo_conn[color_map[0]][color_map[1]]

        #self.emotions = self.collection_color_order.find_one({ 'order': 'group-maxis'})['emotion']

        ## get theme color mapping
        self.color_map = collection_color_map.find_one({'theme': color_theme})['map']

    def dump(self, filename):
        utils.save_pkl_file(self, filename)

    def load(self, filename):
        self = utils.load_pkl_file(filename)


