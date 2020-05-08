import keras
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
    min_level_db = -100
    ref_level_db = 20

    def __init__(self, file_to_int,augment=False,freq_compress='linear'):
        self.file_to_int = file_to_int
        self.Aug = augment
        self.window = np.hanning(self.n_fft)
        self.freq_compress = freq_compress

    def chunker(self,seq, size):
        x = []
        for pos in range(0,len(seq),size):
            x.append(seq[pos:pos + size])
        return x

    def augmentation(self,spec):
        spec = randomAmplitude(spec)
        spec = randomPitchShift(spec)
        spec = randomTimeStretch(spec)
        return spec

    def load_audio_file(self, file_path):

        y, sr = librosa.core.load(file_path, sr=44100, mono=True)
        y = y[np.newaxis,:]
        y = np.concatenate((np.expand_dims(y[:,0],-1),y[:,1:] - self.coef * y[:,:-1]), -1)
        #Noise = False
        #if self.file_to_int.get(file_path) == 0:
        #       Noise = True
        data = self.preprocess_audio(np.squeeze(y,0))
        return data

    def spectrogram(self,y):
        spec = librosa.core.stft(y, n_fft=4096, center=False, hop_length=441,window='hann')
        div = math.sqrt(np.power(self.window,2).sum())
        spec /= div
        return np.power(spec,2)

    def preprocess_audio(self, y):

        spec = self.spectrogram(y)

        if self.Aug:
            spec = self.augmentation(spec)

        if self.freq_compress == "linear":
            spec = self.Interpolate(spec)
        else:
            spec = librosa.feature.melspectrogram(S=spec, n_fft=self.n_fft, hop_length=self.hop_length, sr=self.sample_r,
                                              fmin=self.fmin, fmax=self.fmax, n_mels=self.n_mels,center=self.center)

        spec = librosa.power_to_db(abs(spec))
        spec = np.clip((spec - self.ref_level_db - self.min_level_db) / -self.min_level_db,a_min=0,a_max=1)
        spec = paddingorsampling(spec, self.Time,self.Aug)

        return spec.T

    def Interpolate(self,spec):

        if self.sample_r is not None and self.n_fft is not None:
            min_bin = int(max(0, math.floor(self.n_fft * self.fmin / self.sample_r)))
            max_bin = int(min(self.n_fft - 1, math.ceil(self.n_fft * self.fmax / self.sample_r)))
            spec = spec[min_bin:max_bin,:]
        spec = nn_interpolate(spec,[self.n_mels,spec.shape[1]])
        return spec



class Train_Generator(keras.utils.Sequence):

    def __init__(self, list_files, file_to_int,freq_compress='linear', augment=False, batch_size=32):
        self.data = list_files
        self.batch_size = batch_size
        self.file_to_int = file_to_int
        self.dl = Dataloader(file_to_int,augment,freq_compress=freq_compress)

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, index):

        #self.test_index_set.add(index)
        #self.test_index_list.append(index)
        batch_files = self.data[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = [self.dl.load_audio_file(fpath) for fpath in batch_files]
        batch_data = np.array(batch_data)[:, :, :, np.newaxis]
        batch_labels = [self.file_to_int[fpath] for fpath in batch_files]
        batch_labels = np.array(batch_labels)

        return batch_data, batch_labels

    def on_epoch_end(self):

        #self.test.test_all_indexes(self.test_index_set,self.test_index_list,self.test_indexes)
        shuffle(self.data)
        #self.test_index_list.clear()
        #self.test_index_set.clear()
