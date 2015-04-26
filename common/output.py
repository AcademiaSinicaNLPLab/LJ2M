
import csv

def dump_dict_to_csv(file_name, data):
    w = csv.writer(open(file_name, 'w'))
    for key, val in data.items():
        w.writerow([key, val])

def dump_list_to_csv(file_name, data):
    w = csv.writer(open(file_name, 'w'))
    for row in data:
        w.writerow(row)