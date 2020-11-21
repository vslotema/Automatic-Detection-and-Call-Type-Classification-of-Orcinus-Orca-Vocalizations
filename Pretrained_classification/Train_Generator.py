
import keras
import librosa

from Augment import *
from random import shuffle
import math
import numpy as np


class Dataloader(keras.utils.Sequence):

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

    def __init__(self,noise_files,augment=False,freq_compress='linear',addnoise=False):
        if noise_files is None:
            self.noise_files = []
        else:
            self.noise_files = noise_files
        self.Aug = augment
        self.window = np.hanning(self.n_fft)
        self.freq_compress = freq_compress
        if not augment:
            self.center = True
        self.addnoise = addnoise

    def load_audio_file(self, file_path):
        y, sr = librosa.core.load(file_path, sr=44100, mono=True)
        y = self.preEmphasize(y)
        data = self.preprocess_audio(np.squeeze(y,0),file_path)
        return data

    def preEmphasize(self,y):
        y = y[np.newaxis, :]
        y = np.concatenate((np.expand_dims(y[:, 0], -1), y[:, 1:] - self.coef * y[:, :-1]), -1)
        return y

    def preprocess_audio(self, y,file_path):

        spec = self.spectrogram(y)

        if self.Aug:
            spec = self.augmentation(spec)

        if self.freq_compress == "linear":
            spec = self.Interpolate(spec)
        else:
            spec = self.MelFrequency(spec)

        noise_spec = None
        if self.addnoise:
            #idx = np.random.randint(0,len(self.noise_files)-1)
           # noise_file = self.noise_files[idx]
            spec = self.addRandomNoise(spec,file_path)

        spec = librosa.power_to_db(abs(spec))
        spec = np.clip((spec - self.ref_level_db - self.min_level_db) / -self.min_level_db,a_min=0,a_max=1)
        spec = paddingorsampling(spec, self.Time,self.Aug)

        return spec.T

    def spectrogram(self, y):
        spec = librosa.core.stft(y, n_fft=self.n_fft, center=self.center, hop_length=self.hop_length, window='hann')
        div = math.sqrt(np.power(self.window, 2).sum())
        spec /= div
        return np.power(spec, 2)

    def addRandomNoise(self,spec,noise_file):

        y, sr = librosa.core.load(noise_file, sr=44100, mono=True)
        y = self.preEmphasize(y)
        noise_spec = self.spectrogram(np.squeeze(y, 0))

        sampler = lambda x: np.random.randint(
            0, x, size=(1,), dtype='long'
        ).item()
        noise_spec = sampling(noise_spec, sampler, seq_length=spec.shape[1] * 2)
        noise_spec = randomTimeStretch(noise_spec)
        noise_spec = randomPitchShift(noise_spec)
        if self.freq_compress == "linear":
            noise_spec = self.Interpolate(noise_spec)
        else:
            noise_spec = self.MelFrequency(noise_spec)

        spec = paddingorsampling(spec, 128, True)

        if spec.shape[1] > noise_spec.shape[1]:
            n_repeat = int(math.ceil(spec.shape[1] / noise_spec.shape[1]))
            noise_spec = noise_spec.repeat(n_repeat, 1)
        if spec.shape[1] < noise_spec.shape[1]:
            high = noise_spec.shape[1] - spec.shape[1]
            start = np.random.randint(0, high, size=(1,), dtype='long')
            end = start + spec.shape[1]
            noise_spec_part = noise_spec[:,start[0]:end[0]]
        else:
            noise_spec_part = noise_spec

        snr = np.random.randint(-3, 12, size=(1,)).astype("float64")
        signal_power = spec.sum()
        noise_power = noise_spec_part.sum()

        K = (signal_power / noise_power) * 10 ** (-snr / 10)
        spectrogram_aug = spec + noise_spec_part * K
        return spectrogram_aug

    def augmentation(self,spec):
        spec = randomAmplitude(spec)
        spec = randomPitchShift(spec)
        spec = randomTimeStretch(spec)
        return spec

    def Interpolate(self,spec):

        if self.sample_r is not None and self.n_fft is not None:
            min_bin = int(max(0, math.floor(self.n_fft * self.fmin / self.sample_r)))
            max_bin = int(min(self.n_fft - 1, math.ceil(self.n_fft * self.fmax / self.sample_r)))
            spec = spec[min_bin:max_bin,:]
        spec = nn_interpolate(spec,[self.n_mels,spec.shape[1]])
        return spec

    def MelFrequency(self,spec):
        n_fft = spec.shape[0]
        mel = self._melbank(n_fft)
        spec_m = np.matmul(mel,spec)
        return spec_m

    def _melbank(self,n_fft):
        m_min = self._hz2mel(self.fmin)
        m_max = self._hz2mel(self.fmax)

        m_pts = np.linspace(m_min,m_max,self.n_mels+2)
        f_pts = self._mel2hz(m_pts)

        bins = np.floor(((n_fft - 1) * 2 + 1) * f_pts / self.sample_r).astype("long")

        fb = np.zeros((self.n_mels, n_fft))
        for m in range(1, self.n_mels + 1):
            f_m_minus = bins[m - 1].item()
            f_m = bins[m].item()
            f_m_plus = bins[m + 1].item()

            if f_m_minus != f_m:
                fb[m - 1, f_m_minus:f_m] = (np.arange(f_m_minus, f_m) - f_m_minus).astype("float") / (
                        f_m - f_m_minus
                )
            if f_m != f_m_plus:
                fb[m - 1, f_m:f_m_plus] = (f_m_plus - np.arange(f_m, f_m_plus)).astype("float") / (
                        f_m_plus - f_m
                )

        return fb

    def _hz2mel(self,f):
        return 2595 * np.log10(1 + f / 700)

    def _mel2hz(self,mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def chunker(self,seq, size):
        x = []
        for pos in range(0,len(seq),size):
            x.append(seq[pos:pos + size])
        return x

class Train_Generator(keras.utils.Sequence):

    def __init__(self, split, list_files, file_to_int,freq_compress='linear', augment=False, batch_size=32,add_noise=False):
        self.split = split
        self.data = list_files
        self.batch_size = batch_size
        self.file_to_int = file_to_int
        self.add_noise = add_noise
        if add_noise:
            self.dl= Dataloader(list_files,augment,freq_compress=freq_compress,addnoise=True)
        else:
            self.dl = Dataloader(None,augment,freq_compress=freq_compress)

    def __len__(self):
        if self.split == "train":
            return math.floor(len(self.data) / self.batch_size)
        else:
            return math.ceil(len(self.data) / float(self.batch_size))

    def __getitem__(self, index):

        batch_files = self.data[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = [self.dl.load_audio_file(fpath) for fpath in batch_files]
        batch_data = np.array(batch_data)[:, :, :, np.newaxis]
        batch_labels = [self.file_to_int[fpath] for fpath in batch_files]
        batch_labels = np.array(batch_labels)

        return batch_data, batch_labels

    def on_epoch_end(self):

        shuffle(self.data)