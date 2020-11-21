import numpy as np
import librosa
import os
import glob
import scipy.io.wavfile
from scipy.io import wavfile
from Augment import *
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", type=str,
                help="path for noise files to be augmented")
ap.add_argument("--pitch-shift", type=int,
                help="0 or 1 to apply pitch shifting")
ap.add_argument("--time-stretch", type=int,
                help="0 or 1 to apply time-stretching")
ap.add_argument("--add-noise", type=int,
                help="0 or 1 to add random noise")

ARGS = ap.parse_args()

if __name__ == '__main__':

    data_dir = ARGS.data_dir
    # FILE FORMATE: call/noise-XXX_ID_YEAR_TAPENAME_STARTTIME_ENDTIME.wav
    shift_pitch = ARGS.pitch_shift
    stretch_time = ARGS.time_stretch
    add_noise = ARGS.add_noise

    bins_per_octave = 12
    pitch_pm = 2
    ID = 0
    #startnum = '149817'
    #ending = '150055.wav'

    files = sorted(os.listdir(data_dir))

    # PITCH SHIFTING
    def pitch_gen(data_dir, src_name, dest_name):
        samples, samplerate = librosa.core.load(data_dir + src_name, sr=44100, mono=True)
        y_pitch = samples.copy()
        pitch_change = pitch_pm ** (np.random.uniform(low=math.log2(0.5), high=math.log2(1.5)))
        #print(src_name + ": pitch change = ", pitch_change)
        y_pitch = librosa.effects.pitch_shift(y_pitch.astype('float64'),
                                              sr=44100, n_steps=pitch_change,
                                              bins_per_octave=bins_per_octave)

        librosa.output.write_wav(data_dir + dest_name, y_pitch, sr=44100, norm=True)


    # TIME STRETCHING
    def time_stretch_gen(data_dir, src_name, dest_name):
        samples, samplerate = librosa.core.load(data_dir + src_name, sr=44100, mono=True)
        input_length = len(samples)
        factor = 2 ** np.random.uniform(low=-1, high=1)
        y_stretched = samples.copy()
        y_stretched = librosa.effects.time_stretch(y_stretched.astype('float'), factor)
        #print(src_name + ": stretch factor = ", factor)
        if len(y_stretched) > input_length:
            y_stretched = y_stretched[:input_length]
        else:
            y_stretched = np.pad(y_stretched, (0, max(0, input_length - len(y_stretched))), "constant")

        librosa.output.write_wav(data_dir + dest_name, y_stretched, sr=44100, norm=True)


    # EXTRA NOISE ADDITION
    def add_noise_gen(data_dir, src_name, dest_name, low=-3, high=12):
        samples, samplerate = librosa.core.load(data_dir + src_name, sr=44100, mono=True)
        low =low
        high= high
        y_noise = samples.copy()
        noise_amp = 0.005*np.random.uniform(low=low,high=high)*np.amax(y_noise)
        y_noise = y_noise.astype('float64') + noise_amp * np.random.normal(size=y_noise.shape[0])

        librosa.output.write_wav(data_dir + dest_name, y_noise, sr=44100, norm=True)


    list_augmented = []

    if add_noise == 1:
        for file in files:
            ID = ID+1
            split = file.split('_')
            num = (split[1][:4])
            year = (split[2])
            name = (split[0])
            startnum = (split[4])
            ending = (split[5])
            file_noise_added = 'noise-added_{}'.format(str(ID) + (num)) + '_{}'.format(year) \
                               + '_{}'.format(name) + '-{}_'.format(num) \
                               + '{}_'.format(startnum) + '{}'.format(ending)
            list_augmented.append(ID)
            add_noise_gen(data_dir, file, file_noise_added, low=-3, high=12)

        print('LEN LIST FIRST LOOP ', len(list_augmented))

    if stretch_time == 1:
        print('t S')
        for file in files:
            ID = ID+2
            split = file.split('_')
            num = (split[1][:4])
            year = (split[2])
            name = (split[0])
            startnum = (split[4])
            ending = (split[5])
            file_time_stretched = 'noise-timestretch_{}'.format(str(ID) + (num)) + '_{}'.format(year) \
                                  + '_{}'.format(name) + '-{}_'.format(num) \
                                  + '{}_'.format(startnum) + '{}'.format(ending)
            list_augmented.append(ID)
            time_stretch_gen(data_dir, file, file_time_stretched)

        print('LEN LIST SECOND LOOP ', len(list_augmented))


    if shift_pitch == 1:
        for file in files:
            ID = ID+3
            split = file.split('_')
            num = (split[1][:4])
            year = (split[2])
            name = (split[0])
            startnum = (split[4])
            ending = (split[5])
            list_augmented.append(ID)
            file_pitch_shifted = 'noise-pitchshift_{}'.format(str(ID) + (num)) + '_{}'.format(year) \
                                 + '_{}'.format(name) + '-{}_'.format(num) \
                                 + '{}_'.format(startnum) + '{}'.format(ending)
            pitch_gen(data_dir, file, file_pitch_shifted)

        print(('LEN LIST THIRD LOOP ', len(list_augmented)))

