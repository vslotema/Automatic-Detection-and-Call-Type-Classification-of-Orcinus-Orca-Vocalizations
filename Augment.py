
import numpy as np
import math


def nn_interpolate(spec,new_size):
    old_size = spec.shape
    row_ratio, col_ratio = np.array(new_size)/np.array(old_size)
    row_idx = (np.floor(range(0,int(old_size[0]*row_ratio))/row_ratio)).astype(int)
    col_idx = (np.floor(range(0,int(old_size[1]*col_ratio))/col_ratio)).astype(int)
    final_matrix = spec[row_idx,:][:,col_idx]
    return final_matrix

def nearest(spec,rowsize,colsize):
    ratio_row = spec.shape[0]/rowsize
    ratio_col = spec.shape[1]/colsize
    pos_R = np.empty([rowsize])
    pos_W = np.empty([colsize])
    for i in range(len(pos_R)):
        pos_R[i] = int(i*ratio_row)

    for i in range(len(pos_W)):
        pos_W[i] = int(i*ratio_col)

    new = np.empty([rowsize,colsize],dtype='complex')

    for i in range(rowsize):
        for j in range(colsize):
            new[i,j] = spec[int(pos_R[i]), int(pos_W[j])]

    return new

def scale(spec, factor, dim):
    in_dim = spec.ndim
    if in_dim < 2:
        raise ValueError("Expected spectrogram with size (f t)"
                         ", but got {}".format(spec.size)
                         )
    size = list(spec.shape)
    size[dim] = int(np.round(size[dim] * factor))
    spec = nn_interpolate(spec,size)
    return spec


def randomTimeStretch(spec):
    factor = 2 ** np.random.uniform(low=-1., high=1.)
    scaled = scale(spec, factor, 1)

    if spec.shape[0] != scaled.shape[0]:
        raise ValueError("Expected frequency to stay the same "
                         ", but got {}".format(scaled.shape[0])
                         )
    return scaled


def randomPitchShift(spec):
    factor = 2 ** np.random.uniform(low=math.log2(0.5), high=math.log2(1.5))
    median = np.median(spec)
    size = list(spec.shape)
    scaled = scale(spec, factor, 0)

    if spec.shape[1] != scaled.shape[1]:
        raise ValueError("Expected time to stay the same "
                         ", but got {}".format(scaled.shape[1])
                         )
    if factor > 1:
        out = scaled[: size[0], :]
    else:
        out = np.full(size, median, dtype=spec.dtype)
        new_f_bins = int(np.round(size[0] * factor))
        out[0:new_f_bins, :] = scaled
    return out

def randomAmplitude(spec):
    dyn_change = np.random.randint(low=-6., high=3.)
    spec = spec * (10 ** (dyn_change / 10))
    return spec

def paddingorsampling(spect,seq_length,augment):
    sampler = lambda x: x // 2
    sample_length = spect.shape[1]
    if augment:
        sampler = lambda x: np.random.randint(
                0, x, size=(1,), dtype='long'
                ).item()

    if sample_length < seq_length:
        spect = padding(spect,sampler,seq_length=seq_length)
    else:
        spect = sampling(spect,sampler,seq_length=seq_length)
    return spect

def padding(spect,sampler, seq_length=None, dim = 1):

    sample_length = spect.shape[1]
    start = sampler(seq_length - sample_length) # returns random start number
    end = start + sample_length                                     # random end
    shape = list(spect.shape)                                       # returns a list with similar shape to spect
    shape[1] = seq_length
    padded_spect = np.zeros(shape,dtype='long')

    if dim == 0:
        padded_spect[start:end] = spect
    elif dim == 1:
        padded_spect[:,start:end] = spect
    elif dim == 2:
        padded_spect[:,:, start:end] = spect
    elif dim == 3:
        padded_spect[:,:,:,start:end] = spect
    return padded_spect

def sampling(spect,sampler,seq_length=None, dim =1):

    sample_length = spect.shape[1]
    if sample_length > seq_length:
        start = sampler(sample_length - seq_length)
        end = start + seq_length
        indices= np.arange(start,end,dtype='long')
        return np.take(spect, indices, axis=dim)
    return spect

def addNoise(AudioArr,low=-3,high=12):
    y_noise = AudioArr.copy()
    noise_amp = 0.005*np.random.uniform(low=low,high=high)*np.amax(y_noise)
    y_noise = y_noise.astype('complex') + noise_amp * np.random.normal(size=y_noise.shape[0])
    return y_noise
