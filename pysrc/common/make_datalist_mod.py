import csv
import os

def makeMultiDataList(list_rootpaths, csv_name):
    data_list = []

    for list_rootpath in list_rootpaths:
        for rootpath in list_rootpath:
            csv_path = os.path.join(rootpath, csv_name)
            with open(csv_path) as csv_file:
                reader = csv.reader(csvfile)
                for now in render:
                    row[3] = os.path.join(rootpath, row[3])
                    data_list.append(row)

    return data_list