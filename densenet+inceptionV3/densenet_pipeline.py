import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from Train_Generator import *
import matplotlib.pyplot as plt
import time
import os
from datetime import date
import threading

import tensorflow.keras.backend as K
from keras import optimizers, losses, activations, models
from keras.layers import Dense, Input, Dropout, BatchNormalization, Convolution2D, MaxPooling2D, GlobalMaxPool2D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
from DebugWeights import *
import timeit
import resnet
import multiprocessing
import densenet121
import inception_v3
#import densenet_simple

from scipy import stats

import numpy as np
from keras.models import Model
from keras.layers import Dense
from keras.applications.densenet import DenseNet121


if __name__=='__main__':
    batch_size = 32
    epochs = 50
    # In[2]:
    t = time.localtime()
    ID = time.strftime("%H-%M-%S", t)
    today = date.today()

    dir = str(today) + "_" + str(ID)
    os.mkdir(dir)

    HISTORY_CSV = "{}".format(dir) + "/history_{}.csv".format(ID)
    TEST_FILES_CSV = "{}".format(dir) + "/test_files_{}.csv".format(ID)
    MODEL_STRUCTURE_JSON = "{}".format(dir) + "/model_structure_{}.json".format(ID)
    WEIGHTS_CNN = "{}".format(dir) + "/weights_{}.h5".format(ID)
    BEST_MODEL = "{}".format(dir) + "/best_model_{}.h5".format(ID)
    PLOT = "{}".format(dir) + "/plot_{}.png".format(ID)
    RESULTS_CSV = "{}".format(dir) + "/results_{}.csv".format(ID)
    GRADIENTS_CSV = "{}".format(dir) + "/gradients_{}.csv".format(ID)
    WEIGHTS_CSV = "{}".format(dir) + "/weights_{}.csv".format(ID)


    def append_to_list(path_wav, file_csv):
        list_f = []
        csv = pd.read_csv(file_csv)
        for file in csv.file_name.values:
            list_f.append(path_wav + file)
        return list_f


    tr_files = []
    val_files = []

    path = '/Users/charlottekruss/THESIS/csv/' #'ORCASPOT_csv/'

   # DLFD = path + "DLFD/"
   # DLFD_lab = path + "DLFD/lab/"
   # tr_files += append_to_list(DLFD, DLFD_lab + "new_train_label.csv")
   # val_files += append_to_list(DLFD, DLFD_lab + "new_val_label.csv")

    #OAC_wav = 'Orchive/extract/' #'Orchive/extract/'#
    #OAC_lab = path + "OAC/"
    #tr_files += append_to_list(OAC_wav, OAC_lab + "new_train_label_cor.csv")
    #val_files += append_to_list(OAC_wav, OAC_lab + "new_val_label_cor.csv")
    #print('VAL FILES 1', len(val_files))

    DeepAL_wav = '/Users/charlottekruss/THESIS/DeepAL_ComParE/ComParE2019_OrcaActivity/wav/'#'wav/'
    DeepAL_lab = path + 'DeepAL/'
    tr_files += append_to_list(DeepAL_wav, DeepAL_lab + "DeepALnew_train_label.csv")
    print('TR FILES 2 ', len(tr_files))
    val_files += append_to_list(DeepAL_wav, DeepAL_lab + "DeepALnew_val_label.csv")
    print('VAL FILES 2 ', len(val_files))

    Audible_wav = 'Orchive/audible/'
    Audible_lab = path + 'ORCASPOT_csv/audible/'
    tr_files += append_to_list(Audible_wav, Audible_lab + "train_audible.csv")
    print('TR FILES 2 ', len(tr_files))
    val_files += append_to_list(Audible_wav, Audible_lab + "val_audible.csv")
    print('VAL FILES 2 ', len(val_files))

    #AEOTD_wav = path + "AEOTD/AEOTD/"
    #AEOTD_lab = path + "AEOTD/AEOTD_label/"
    #tr_files += append_to_list(AEOTD_wav, AEOTD_lab + "train_label_new.csv")
    #val_files += append_to_list(AEOTD_wav, AEOTD_lab + "val_label_new.csv")
    #print('VAL FILES 3 ', len(val_files))


    def file_to_label(file_to_label, wav_path, csv_path):

        csv = pd.read_csv(csv_path)

        file_to_label.update({wav_path + k: v for k, v in zip(csv.file_name.values, csv.label.values)})
        return file_to_label


    file_to_label_train = {}
    #file_to_label(file_to_label_train, DLFD, DLFD_lab + "new_train_label.csv")
    #print('TRAIN 1',len(file_to_label_train))
    #file_to_label(file_to_label_train, OAC_wav, OAC_lab + "new_train_label.csv")
    #print('TRAIN 2',len(file_to_label_train))
    file_to_label(file_to_label_train, DeepAL_wav, DeepAL_lab + "DeepALnew_train_label.csv")
    print('FILE TO LABEL TRAIN 3', len(file_to_label_train))
    #file_to_label(file_to_label_train, AEOTD_wav, AEOTD_lab + "train_label_new.csv")
    #print('TRAIN 4', len(file_to_label_train))
    file_to_label(file_to_label_train, Audible_wav, Audible_lab + "train_audible.csv")
    print('TRAIN 5', len(file_to_label_train))

    file_to_label_val = {}
    #file_to_label(file_to_label_val, AEOTD_wav, AEOTD_lab + "val_label_man.csv")
   # print("VAL LABEL 1", len(file_to_label_val))
    #file_to_label(file_to_label_val, DLFD, DLFD_lab + "new_val_label.csv")
    #print("VAL LABEL 2", len(file_to_label_val))
    #file_to_label(file_to_label_val, OAC_wav, OAC_lab + "new_val_label.csv")
    #print("VAL LABEL 1", len(file_to_label_val))
    file_to_label(file_to_label_val, DeepAL_wav, DeepAL_lab + "DeepALnew_val_label.csv")
    print("FILE TO LABEL VAL 2", len(file_to_label_val))
    #file_to_label(file_to_label_val, AEOTD_wav, AEOTD_lab + "val_label_new.csv")
    #print("VAL LABEL 3", len(file_to_label_val))
    file_to_label(file_to_label_val, Audible_wav, Audible_lab + "val_audible.csv")
    print("VAL LABEL 3", len(file_to_label_val))


    list_labels = ["noise", "orca"]  # Unique labels orca and noise
    print('list_labels ', list_labels)

    label_to_int = {k: v for v, k in enumerate(list_labels)}
    print('label_to_int ', label_to_int)

    file_to_int_train = {k: label_to_int[v] for k, v in file_to_label_train.items()}
    file_to_int_val = {k: label_to_int[v] for k, v in file_to_label_val.items()}

    print('TRAIN ',len(file_to_int_train))
    print('VAL ', len(file_to_int_val))


    def extract_layer_output(model, layer_name, input_data):

      layer_output_fn = K.function([model.layers[0].input],[model.get_layer(layer_name).output])
      layer_output = layer_output_fn([input_data])
     # layer_output.shape is (num_units, num_timesteps)

      return layer_output[0]

    '''Build a simple densenet model '''
    #def build_simple_densenet():
    #    dense_block_size = 3
    #    layers_in_block = 4
    #    growth_rate = 12
    #    classes = 1
    #    model = densenet_simple.dense_net(growth_rate * 2, growth_rate, classes, dense_block_size, layers_in_block)
    #    return model

    '''Build an InceptionV3 model '''
    #base_model = inception_v3.InceptionV3(input_shape=(128, 256, 1),
    #                                    weights=None,
    #                                    include_top=False,
    #                                    pooling='avg')

    '''Build a Densenet121 (optionally 169|201) model '''
    base_model = densenet121.DenseNet121(input_shape=(128, 256, 1),
                                            weights=None,
                                            include_top=False,
                                            pooling='avg')
                                        
    current_layer = base_model.output
    predictions = Dense(1, activation='sigmoid')(current_layer)
    model = Model(inputs=base_model.input, outputs=predictions)

    #print(model.summary())
    #print(model.predict(np.random.random((10, 128, 256, 1))))

    #model = get_model_mel()
    #model = resnet.ResnetBuilder.build_resnet_18((128, 256, 1), 1)
    #model = resnet.ResnetBuilder.build_resnet_50((128, 256, 1), 1)

    opt = optimizers.Adam(lr=0.00001, beta_1=0.5, beta_2=0.999, amsgrad=False)
    #opt = keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["acc"])
    model.summary()


    # In[]:
    dl = Dataloader(file_to_int_val)
    gradcsv= pd.read_csv(DeepAL_lab+"DeepALnew_val_label.csv")
    T = []
    L = []
    for i in range(0,10):
        if gradcsv.label.values[i] == "noise":
            L.append(0)
        else:
            L.append(1)
        x = dl.load_audio_file(DeepAL_wav+gradcsv.file_name.values[i])
        T.append(x)
    train_files = np.array(T)[:,:, :, np.newaxis]

    #my_debug_weights = MyDebugWeights(train_files, L, WEIGHTS_CSV, GRADIENTS_CSV)

    # In[]:
    training_generator = Train_Generator(tr_files, file_to_int_train,augment=True,batch_size=batch_size)
    validation_generator = Train_Generator(val_files, file_to_int_val,augment=False,batch_size=batch_size)

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, mode='max',
                                                           patience=4, min_delta=1e-3)

    tb = TensorBoard(log_dir='./Graph/{}'.format(ID), histogram_freq=0, write_graph=True, write_images=True)

    earlystop = EarlyStopping(monitor='val_acc', patience=10, mode='max')
    mcheckpoint= ModelCheckpoint(BEST_MODEL,monitor='val_acc',save_best_only=True,mode='max')

    start = timeit.default_timer()
    H = model.fit_generator(generator=training_generator, epochs=epochs, steps_per_epoch=len(tr_files) // batch_size,
                            validation_data=validation_generator, validation_steps=len(val_files) // batch_size,use_multiprocessing=False,
                            workers=multiprocessing.cpu_count(),verbose=1,
                            callbacks=[tb, earlystop,mcheckpoint,reduce_lr])

    #epoch_stop = earlystop.stopped_epoch
    num_epochs_run = len(H.history['acc'])
    stop = timeit.default_timer()
    print('Time: ', stop - start)


    # In[]:

    model_structure = model.to_json()# convert the NN into JSON
    f = Path(MODEL_STRUCTURE_JSON) # write the JSON data into a text file
    f.write_text(model_structure) # Pass in the data that we want to write into the file.

    model.save_weights(WEIGHTS_CNN)

    pd.DataFrame.from_dict(H.history).to_csv(HISTORY_CSV,index=False)


    # In[19]:

    # plot the training loss and accuracy
    N = num_epochs_run
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(PLOT)
