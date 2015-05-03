
import os
import csv
import time

def create_folder_with_time(prefix):
    t = time.strftime('%y%m%d%H%M%S', time.localtime(time.time()))
    name = '%s_%s' % (prefix, t)

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