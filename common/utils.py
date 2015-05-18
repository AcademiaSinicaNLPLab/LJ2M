
import cPickle as pickle
import logging
import numpy as np

def sigmoid(x):
    neg_x = map(lambda v: v*(-1), x)
    return 1 / (1 + np.exp(neg_x))

def save_pkl_file(clz, filename):
    try:
        pickle.dump(clz, open(filename, "w"))
    except ValueError:
        logging.error("failed to dump %s" % (filename))

def load_pkl_file(filename):
    try:
        pkl = pickle.load(open(filename, "r"))
        return pkl
    except ValueError:
        logging.error("failed to load %s" % (filename))

def get_unique_list_diff(a, b):
    return list(set(a) - set(b))

def parse_range(astr):
    result = set()
    for part in astr.split(','):
        x = part.split('-')
        result.update(range(int(x[0]), int(x[-1]) + 1))
    return sorted(result)

def parse_list(astr):
    result = set()
    for part in astr.split(','):
        result.add(float(part))
    return sorted(result)

def test_writable(file_path):
    writable = True
    try:
        filehandle = open(file_path, 'w')
    except IOError:
        writable = False
        
    filehandle.close()
    return writable

def read_parameter_file(file_path):
    """
        this is a temporary solution
        ToDo: generalize
    """
    param_dict = {}
    with open(file_path) as f:
        for line in f:

            toks = line.strip().split(',')
            # only work in python 2.x
            toks[1] = float(toks[1].translate(None, '"()'))
            toks[2] = float(toks[2].translate(None, '"()'))

            param_dict[toks[0]] = toks[1], toks[2]

    return param_dict

def rgba_to_rgb(rgba, bg=(255,255,255)):
    ''' 
    convert RGBA to RGB upder different background
    '''
    bg_r, bg_g, bg_b = bg
    if type(rgba) not in (tuple, list) or len(rgba) != 4:
        raise TypeError('rgba is a tuple or list containing four values')

    r, g, b, a = rgba
    return tuple(map(lambda x:int(x), [(1-a)*bg_r + a*r, (1-a)*bg_g + a*g, (1-a)*bg_b + a*b]))  
