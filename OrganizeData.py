import pandas as pd
import os
import re

def append_to_list(path_wav, file_csv):
    list_f = []
    print("*** append_to_list ***")
    print("file_csv ", file_csv)
    csv = pd.read_csv(file_csv)
    print("csv ", csv)
    for file in csv.file_name.values:
        list_f.append(path_wav + file)
    return list_f

def file_to_label(file_to_label,wav_path,csv_path):
    csv = pd.read_csv(csv_path)
    file_to_label.update({wav_path + k: v for k, v in zip(csv.file_name.values,csv.label.values)})
    return file_to_label


def findpathwav(line):
    split = line.split("/")
    return split[len(split)-2]



def findcsv(tvt,data_dir):
    print("*** findcsv ***")
    print("tvt ", tvt)
    print("data_dir ", data_dir)
    files = []
    ftl = {}
    with open(data_dir + tvt,'r') as fp:
        for c in fp:
            csvpath = data_dir + c.replace("\n","")
            path_wav = data_dir + findpathwav(csvpath) + "/"
            files += append_to_list(path_wav,csvpath)
            file_to_label(ftl,path_wav,csvpath)
    fp.close()
    print("Files {} ".format(tvt),files)
    print("File_to_label {} ", ftl)
    return files, ftl

