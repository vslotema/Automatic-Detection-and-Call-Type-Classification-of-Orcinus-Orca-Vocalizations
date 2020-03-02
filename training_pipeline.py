# coding: utf-8

# In[1]:


import glob
from random import shuffle

import librosa
import numpy as np
import pandas as pd
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input, Dropout, BatchNormalization, Convolution2D, MaxPooling2D, GlobalMaxPool2D
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
from Augment import *
import matplotlib as plt




# In[2]:


#input_length = 16000 * 5

batch_size = 32
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

# def audio_norm(data):
#
#     max_data = np.max(data)
#     min_data = np.min(data)
#     data = (data-min_data)/(max_data-min_data+0.0001)
#     return data-0.5
# Code based on:
# huseinzol05 (2019) Sound Augmentation Librosa [source code] https://www.kaggle.com/huseinzol05/sound-augmentation-librosa?fbclid=IwAR0n3w1pEd8iAJZoE_0UhcNKjtvtAKP3sNNa5tNzWAG1zN9IKEwwRDDrGKk



def preprocess_audio_mel_T(audio, sample_rate=44100, window_size=20,  # log_specgram
                           step_size=10, eps=1e-10):

  #  mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
    stft = librosa.stft(audio.astype('float'), n_fft=n_fft, hop_length=hop_length, center=False)
    mels = librosa.feature.melspectrogram(audio, sr=sample_r, S=stft, fmin=fmin, fmax=fmax, n_mels=bins)
    mels = librosa.util.normalize(mels)
    mels = paddingorsampling(mels, Time)
    mel_db = librosa.power_to_db(abs(mels))
    #mel_db = (librosa.power_to_db(mels, ref=np.max) + 40) / 40

    return mel_db.T


def load_audio_file(file_path):
   # print('FILE PATH VALUE', file_to_int.get(file_path))
    x, sr= librosa.core.load(file_path, sr=sample_r, mono=mono)  # , sr=16000
    x = librosa.effects.preemphasis(x, coef=coef)
    x = randomAmplitude(x)
    x = randomPitchShift(x)
    x = randomTimeStretch(x)

    if file_to_int.get(file_path) == 0:
        x = addNoise(x)



 #   if len(data) > input_length:
#
#        max_offset = len(data) - input_length
#
#        offset = np.random.randint(max_offset)
#
#        data = data[offset:(input_length + offset)]


#    else:
#        if input_length > len(data):
#            max_offset = input_length - len(data)

#            offset = np.random.randint(max_offset)
#        else:
#            offset = 0

 #       data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

    data = preprocess_audio_mel_T(x,sample_rate=sr)
    return data


# In[3]:
wav_path = "C:\\myProjects\\THESIS\\DeepAL_ComParE\\DeepAL_ComParE\\ComParE2019_OrcaActivity\\wav\\"
data_orig = pd.read_csv("C:\\myProjects\\THESIS\\DeepAL_ComParE\\DeepAL_ComParE\\ComParE2019_OrcaActivity\\lab\\DeepAL_ComParE.csv")
labels = data_orig['label']

train, test = np.split(data_orig.sample(frac=1), [int(.8*len(data_orig))])

#train_files = glob.glob("C:\\myProjects\\THESIS\\DeepAL_ComParE\\DeepAL_ComParE\\ComParE2019_OrcaActivity\\wav\\*.wav")
#test_files = glob.glob("C:\\myProjects\\THESIS\\DeepAL_ComParE\\DeepAL_ComParE\\ComParE2019_OrcaActivity\\wav\\*.wav")
#train_labels = pd.read_csv("C:\\myProjects\\THESIS\\DeepAL_ComParE\\DeepAL_ComParE\\ComParE2019_OrcaActivity\\lab\\DeepAL_ComParE.csv")

# In[4]:

#print(train_files)
file_to_label = {wav_path + k: v for k, v in zip(train.file_name.values, train.label.values)}

# In[5]:


# file_to_label


# In[7]:

#
# data_base = load_audio_file(train_files[0])
# fig = plt.figure(figsize=(14, 8))
# plt.title('Raw wave : %s ' % (file_to_label[train_files[0]]))
# plt.ylabel('Amplitude')
# plt.plot(np.linspace(0, 1, input_length), data_base)
# plt.show()


# In[8]:
list_labels = sorted(list(set(train.label.values))) # Unique labels orca and noise
print('list_labels ', list_labels)

# In[9]:

label_to_int = {k: v for v, k in enumerate(list_labels)}
print('label_to_int ', label_to_int)

# In[10]:
int_to_label = {v: k for k, v in label_to_int.items()}
print('int to label ',int_to_label)

# In[11]:

file_to_int = {k: label_to_int[v] for k, v in file_to_label.items()}
print('file to int', file_to_int.get(wav_path+train.file_name[0]))


# In[12]:


def get_model_mel():
    nclass = len(list_labels)
    inp = Input(shape=(128, 256, 1))
    norm_inp = BatchNormalization()(inp)
    img_1 = Convolution2D(16, kernel_size=(3, 7), activation=activations.relu)(norm_inp)
    img_1 = Convolution2D(16, kernel_size=(3, 7), activation=activations.relu)(img_1)
    img_1 = MaxPooling2D(pool_size=(3, 7))(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu)(img_1)
    img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu)(img_1)
    img_1 = MaxPooling2D(pool_size=(3, 3))(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution2D(128, kernel_size=3, activation=activations.relu)(img_1)
    img_1 = GlobalMaxPool2D()(img_1)
    img_1 = Dropout(rate=0.1)(img_1)

    dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(img_1))
    dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(dense_1))
    dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam()

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model



# In[13]:


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


# In[14]:


def train_generator(list_files, batch_size=batch_size):
    while True:
        shuffle(list_files)
        for batch_files in chunker(list_files, size=batch_size):
            batch_data = [load_audio_file(fpath) for fpath in batch_files]
            batch_data = np.array(batch_data)[:, :, :, np.newaxis]
            batch_labels = [file_to_int[fpath] for fpath in batch_files]
            batch_labels = np.array(batch_labels)

            yield batch_data, batch_labels


# In[15]:
tr_files, val_files = train_test_split(sorted(wav_path+train.file_name.values), test_size=0.1, random_state=42)
print(wav_path+train.file_name.values)
print('length train files ', len(tr_files))
print('length val_files ', len(val_files))

# In[16]:


model = get_model_mel()
# model.load_weights("baseline_cnn.h5")

# In[17]:


History = model.fit_generator(train_generator(tr_files), steps_per_epoch=len(tr_files) // batch_size, epochs=20,
                            validation_data=train_generator(val_files), validation_steps=len(val_files) // batch_size,
                            use_multiprocessing=False, workers=8, max_queue_size=60,
                            callbacks=[ModelCheckpoint("weights_cnn_mel.h5", monitor="val_acc", save_best_only=True),
                                    EarlyStopping(patience=5, monitor="val_acc")])

print('History keys ', History.history.keys())

# In[18]:

model_structure = model.to_json()# convert the NN into JSON
f = Path("model_structure.json") # write the JSON data into a text file
f.write_text(model_structure) # Pass in the data that we want to write into the file.

model.save_weights("weights_cnn_mel.h5")


# model.save_weights("baseline_cnn.h5")
#model.load_weights("baseline_cnn_mel.h5")

# In[19]:

# plot the training loss and accuracy
N = np.arange(0, 20)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, History.history["loss"], label="train_loss")
plt.plot(N, History.history["val_loss"], label="val_loss")
plt.plot(N, History.history["acc"], label="train_acc")
plt.plot(N, History.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('training_loss_01.png')



# In[20]:
bag = 3

array_preds = 0

for i in tqdm(range(bag)):

    list_preds = []

    for batch_files in tqdm(chunker(test, size=batch_size), total=len(test) // batch_size):
        batch_data = [load_audio_file(fpath) for fpath in batch_files]
        batch_data = np.array(batch_data)[:, :, :, np.newaxis]
        preds = model.predict(batch_data).tolist()
        list_preds += preds

    # In[21]:

    array_preds += np.array(list_preds) / bag

# In[22]:


list_labels = np.array(list_labels)

# In[30]:


top_3 = list_labels[
    np.argsort(-array_preds, axis=1)[:, :3]]  # https://www.kaggle.com/inversion/freesound-starter-kernel
pred_labels = [' '.join(list(x)) for x in top_3]

# In[31]:


df = pd.DataFrame(test, columns=["file_name"])
df['label'] = pred_labels

# In[32]:


df['file_name'] = df.fname.apply(lambda x: x.split("/")[-1])

# In[33]:


df.to_csv("baseline_mel_bigger.csv", index=False)
