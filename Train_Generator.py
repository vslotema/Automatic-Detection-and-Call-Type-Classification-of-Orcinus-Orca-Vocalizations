import threading
import keras
import numpy as np
import librosa
from Augment import *
from random import shuffle
from Tests import *
import math


class Dataloader():
    n_mels = 256
    sample_r = 44100
    coef = 0.98
    mono = True
    n_fft = 4096
    hop_length = 441
    center = False
    fmin = 500
    fmax = 10000
    Time = 128
    bins = 256
    min_level_db = -100
    ref_level_db = 20

    def __init__(self, file_to_int,augment=False):
        self.file_to_int = file_to_int
        self.Aug = augment


    def chunker(self,seq, size):
        x = []
        for pos in range(0,len(seq),size):
            x.append(seq[pos:pos + size])
        return x

    def augmentation(self,x,file_path):
        x = randomAmplitude(x)
        x = randomPitchShift(x)
        x = randomTimeStretch(x)
        if self.file_to_int.get(file_path) == 0:
               x = addNoise(x)
        return x


    def load_audio_file(self, file_path):
        # print('FILE PATH VALUE', file_to_int.get(file_path))'
        #curr = "THREAD {}".format(threading.current_thread().getName())
        #print(curr)
        #print(file_path)
        x, sr = librosa.core.load(file_path, sr=44100, mono=True)  # , sr=16000
        x = librosa.effects.preemphasis(x, coef=0.98)
        if self.Aug:
            x = self.augmentation(x,file_path)

        data = self.preprocess_audio_mel_T(x, sample_rate=sr)
        return data

    def preprocess_audio_mel_T(self, audio, sample_rate=44100, window_size=20,  # log_specgram
                               step_size=10, eps=1e-10):
        #  mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
       # stft = librosa.stft(audio.astype('float'), n_fft=self.n_fft, hop_length=self.hop_length, window = 'hann', center=False)

        #print("stft shape ", stft.shape)
        mels = librosa.feature.melspectrogram(audio, n_fft=self.n_fft, hop_length=self.hop_length, sr=self.sample_r,
                                              fmin=self.fmin, fmax=self.fmax, n_mels=self.bins)
        mel_db = librosa.power_to_db(mels)
        mel_db02 = np.clip((mel_db - self.ref_level_db - self.min_level_db) / -self.min_level_db,a_min=0,a_max=1)
        mel_db03 = paddingorsampling(mel_db02, self.Time)

        #(spec - self.ref_level_db - self.min_level_db) / -self.min_level_db, 0, 1

        return mel_db03.T

    def interpolate(self, spec):
        n_fft = (spec.shape[2] - 1) * 2

        if self.sample_r is not None and n_fft is not None:
            min_bin = int(max(0, math.floor(n_fft * self.fmin / self.sample_r)))
            max_bin = int(min(n_fft - 1, math.ceil(n_fft * self.fmax / self.sample_r)))
            spec = spec[:, :, min_bin:max_bin]
        return spec


class Train_Generator(keras.utils.Sequence):

    def __init__(self, list_files, file_to_int, augment=False, batch_size=32):
        self.test = SimpleTest()
        self.test_index_set = set()
        self.test_index_list = []
        self.test_indexes = np.arange(int(np.ceil(len(list_files)//batch_size))+1)
        self.data = list_files
        self.batch_size = batch_size
        #self.lock = threading.Lock()
        self.file_to_int = file_to_int
        self.dl = Dataloader(file_to_int,augment)
       # self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, index):
        #self.lock.acquire()
        #try:
           # curr = "THREAD {}".format(threading.current_thread().getName()) + " INDEX {}".format(str(index))
           # print(curr)
        self.test_index_set.add(index)
        self.test_index_list.append(index)
        batch_files = self.data[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = [self.dl.load_audio_file(fpath) for fpath in batch_files]
        batch_data = np.array(batch_data)[:, :, :, np.newaxis]
        batch_labels = [self.file_to_int[fpath] for fpath in batch_files]
        batch_labels = np.array(batch_labels)

        #finally:
        #    self.lock.release()
        return batch_data, batch_labels

    # def __next__(self):

    #    batch_data, batch_labels = self.__getitem__(self.i)
    #   return batch_data, batch_labels

    def on_epoch_end(self):

        self.test.test_all_indexes(self.test_index_set,self.test_index_list,self.test_indexes)
        shuffle(self.data)
        self.test_index_list.clear()
        self.test_index_set.clear()
