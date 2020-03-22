import librosa
import numpy as np
from PIL import Image
import math
# Intensity Augmentation
def flipsizes(size, left, right):
    print("left ", left)
    print("right ", right)
    size[0] = left
    size[1] = right
    print("size 0 ", size[0])
    print("size 1 ", size[1])
    return size


def scale(spec, factor, dim):
    print("ndim ", spec.ndim)
    in_dim = spec.ndim
    if in_dim < 2:
        raise ValueError("Expected spectrogram with size (f t)"
                         ", but got {}".format(spec.size)
                         )
    size = list(spec.shape)
    print("size ", size)
    size[dim] = int(np.round(size[dim] * factor))
    print("size dim {}".format(dim), size[dim])

    size = flipsizes(size, size[1], size[0])

    print("size after ", size)
    spec = np.array(Image.fromarray(spec).resize(size, resample=Image.NEAREST))
    print("scaled spec shape ", spec.shape)
    return spec


def randomTimeStretch(spec):
    print("AUGMENT 2")
    print("Spec shape ", spec.shape)
    factor = 2 ** np.random.uniform(low=-1, high=1)
    scaled = scale(spec, factor, 1)

    if spec.shape[0] != scaled.shape[0]:
        raise ValueError("Expected frequency to stay the same "
                         ", but got {}".format(scaled.shape[0])
                         )
    return scaled


def randomPitchShift(spec):
    print("AUGMENT 1")
    print("spec shape ", spec.shape)
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
        print("out before ", out.shape)
        new_f_bins = int(np.round(size[0] * factor))
        out[0:new_f_bins, :] = scaled
        print("out after ", out.shape)
    return out

def randomAmplitude(spec):
    print(spec[:50])
    dyn_change = np.random.randint(low=-6, high=3)
    spec = spec * (10 ** (dyn_change / 10))
    print(spec[:50])
    return spec

def paddingorsampling(spect,seq_length):
    sample_length = spect.shape[1]
    if sample_length < seq_length:
        spect = padding(spect,seq_length=seq_length)
    else:
        spect = sampling(spect,seq_length=seq_length)
    return spect

def padding(spect,seq_length=None, dim = 1):
    sampler = lambda x: x // 2
    sample_length = spect.shape[1]

    start = sampler(seq_length - sample_length) # returns random start number
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
    sampler = lambda x: x // 2
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
    y_noise = y_noise.astype('float64') + noise_amp * np.random.normal(size=y_noise.shape[0])
    return y_noise
