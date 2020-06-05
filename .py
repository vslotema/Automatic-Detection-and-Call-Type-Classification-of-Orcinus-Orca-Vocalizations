import pandas as pd
import re
import os


def append_to_list(path_wav, file_csv):
    list_f = []
    csv = pd.read_csv(file_csv)
    for file in csv.file_name.values:
        if file not in list_f:
            list_f.append(path_wav + file)
    return list_f


def file_to_label(files,wav_path, csv_path):
    csv = pd.read_csv(csv_path)
    file_to_label = {}
    file_to_label.update({wav_path + k: v for k, v in zip(csv.file_name.values, csv.label.values)})
    return file_to_label


def findpathwav(line):
    split = line.split("/")
    path = ""
    for i in range(len(split) - 1):
        path += split[i] + "/"
        if os.name == 'nt':
            path += split[i] + "\\"
    print("path ", path)
    return path

def findcsv(tvt, data_dir):
    files = []
    print("data dir ", data_dir)
    with open(data_dir + tvt, 'rb') as fp:
        lines = [l.decode('utf8', 'ignore') for l in fp.readlines()]
        for i in lines:
            print("i ", i)
            csvpath = data_dir + i
            if re.findall("/", i):
                path_wav = data_dir + findpathwav(i)
            else:
                path_wav = data_dir

            files += append_to_list(path_wav, csvpath)
            ftl = file_to_label(files,path_wav, csvpath)
    fp.close()
    return files, ftl
