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


def file_to_label(file_to_label,wav_path, csv_path):
    csv = pd.read_csv(csv_path)
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
    ftl = {}

    with open(data_dir + tvt, 'rb') as fp:
        lines = [l.decode('utf8', 'ignore') for l in fp.readlines()]
        for i in lines:
            print("i ")
            csvpath = data_dir + i.replace("\n","")
            if re.findall("/", i):
                path_wav = data_dir + findpathwav(i)
            else:
                path_wav = data_dir

            files += append_to_list(path_wav, csvpath)
            ftl = file_to_label(ftl,path_wav, csvpath)
    fp.close()
    return files, ftl

def unique(list):
    unique = []
    for i in list:
        if i not in unique:
            unique.append(i)
    return sorted(unique)

def getUniqueLabels(data_dir):
    tvt = ["train","val","test"]
    u_labels = []
    for i in tvt:
        with open(data_dir + i, 'rb') as fp:
            lines = [l.decode('utf8', 'ignore') for l in fp.readlines()]
            for i in lines:
                i = i.replace("\n","")
                csvpath = data_dir + i
                df = pd.read_csv(csvpath)
                u_labels += df.label.values.tolist()
    u_labels = unique(u_labels)
    return u_labels
