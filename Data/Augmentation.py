# Code based on:
# huseinzol05 (2019) Sound Augmentation Librosa [source code] https://www.kaggle.com/huseinzol05/sound-augmentation-librosa?fbclid=IwAR0n3w1pEd8iAJZoE_0UhcNKjtvtAKP3sNNa5tNzWAG1zN9IKEwwRDDrGKk

import librosa
import numpy as np

# Intensity Augmentation
def randomAmplitude(audioArr,low=-6,high=3):
    amp = audioArr.copy()
 #   print(amp[:50])
    dyn_change = np.random.uniform(low=low,high=high)
  #  print("dyn_change = ",dyn_change)
    amp = amp * dyn_change
 #   print(amp[:50])
    return amp

# Pitch Augmentation
def randomPitchShift(audioArr,low=0.5,high=1.5,sr=44100):
    pitch_shift = audioArr.copy()
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change =  pitch_pm * 2*(np.random.uniform(low=low,high=high))
  #  print("pitch_change = ",pitch_change)
    pitch_shift = librosa.effects.pitch_shift(pitch_shift, sr, n_steps=pitch_change, bins_per_octave=bins_per_octave)
    return pitch_shift

# Time augmentation
def randomTimeStretch(audioArr,low=0.5,high=2.0):
    input_length = len(audioArr)
    streching = audioArr.copy()
    stretch = np.random.uniform(low = low,high = high)
    streching = librosa.effects.time_stretch(streching.astype('float'), stretch)
    if len(streching) > input_length:
        streching = streching[:input_length]
    else:
        streching = np.pad(streching, (0, max(0, input_length - len(streching))), "constant")
    return streching

def paddingorsampling(spect,seq_length):
    sample_length = spect.shape[1]
    if sample_length < seq_length:
        spect = padding(spect,seq_length=seq_length)
    else:
        spect = sampling(spect,seq_length=seq_length)
    return spect

def padding(spect,seq_length=None, dim = 1):
    sample_length = spect.shape[1]

    start = np.random.randint(low=0,high=(seq_length-sample_length)) # returns random start number
    end = start + sample_length                                     # random end
    shape = list(spect.shape)                                       # returns a list with similar shape to spect
    shape[1] = seq_length
    padded_spect = np.zeros(shape,dtype='float')

    if dim == 0:
        padded_spect[start:end] = spect.real
    elif dim == 1:
        padded_spect[:,start:end] = spect.real
    elif dim == 2:
        padded_spect[:,:, start:end] = spect.real
    elif dim == 3:
        padded_spect[:,:,:,start:end] = spect.real
    return padded_spect


def sampling(spect,seq_length=None, dim =1):
    sample_length = spect.shape[1]
    if sample_length > seq_length:
        start = np.random.randint(low=0,high=(sample_length-seq_length))
        end = start + seq_length
        indices= np.arange(start,end,dtype='long')
        return np.take(spect, indices, axis=dim)
    return spect

def addNoise(AudioArr,low=-3,high=12):
    y_noise = AudioArr.copy()
    noise_amp = 0.005*np.random.uniform(low=low,high=high)*np.amax(y_noise)
    y_noise = y_noise.astype('float64') + noise_amp * np.random.normal(size=y_noise.shape[0])
    return y_noise
