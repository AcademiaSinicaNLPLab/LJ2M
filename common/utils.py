
import pickle
import logging

def save_pkl_file(clz, filename):
    try:
        pickle.dump(clz, open(filename, "w"))
    except ValueError:
        logging.error("failed to dump %s" % (filename))

def load_pkl_file(filename):
    try:
        return pickle.load(open(filename, "r"))
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
    