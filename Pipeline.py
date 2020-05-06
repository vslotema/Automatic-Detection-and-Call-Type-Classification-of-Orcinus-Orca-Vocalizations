# In[0]:
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split

from Train_Generator import *
import matplotlib.pyplot as plt
import time
import os
from datetime import date
import threading

from keras import optimizers, losses, activations, models
from keras.layers import Dense, Input, Dropout, BatchNormalization, Convolution2D, MaxPooling2D, GlobalMaxPool2D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,  ModelCheckpoint
#from DebugWeights import *
import timeit
import resnet
import multiprocessing


from scipy import stats


    # In[2]:
if __name__=='__main__':
    batch_size = 1
    epochs = 1
    # In[2]:
    t = time.localtime()
    ID = time.strftime("%H-%M-%S", t)
    today = date.today()

    dir = str(today) + "_" + str(ID)
    os.mkdir(dir)

    HISTORY_CSV = "{}".format(dir) + "\\history_{}.csv".format(ID)
    TEST_FILES_CSV = "{}".format(dir) + "\\test_files_{}.csv".format(ID)
    MODEL_STRUCTURE_JSON = "{}".format(dir) + "\\model_structure_{}.json".format(ID)
    WEIGHTS_CNN = "{}".format(dir) + "\\weights_{}.h5".format(ID)
    BEST_MODEL = "{}".format(dir) + "\\best_model_{}.h5".format(ID)
    PLOT = "{}".format(dir) + "\\plot_{}.png".format(ID)
    RESULTS_CSV = "{}".format(dir) + "\\results_{}.csv".format(ID)
    GRADIENTS_CSV = "{}".format(dir) + "\\gradients_{}.csv".format(ID)
    WEIGHTS_CSV = "{}".format(dir) + "\\weights_{}.csv".format(ID)


    # In[]:
    def append_to_list(path_wav,file_csv):
        list_f = []
        csv = pd.read_csv(file_csv)
        for file in csv.file_name.values:
            list_f.append(path_wav+file)
        return list_f

    tr_files = []
    val_files = []

    #path ="C:\\myProjects\\THESIS\\csv\\ORCASPOT_csv\\"
    path = "C:\\myProjects\\THESIS\\ORCA-SPOT\\testdata\\"
    tr_files += append_to_list(path,path + "train_lab.csv")
    val_files+= append_to_list(path,path + "val_lab.csv")

   # AEOTD_wav = path + "AEOTD\\AEOTD\\"
   # AEOTD_lab= path + "AEOTD\\AEOTD_label\\"
  #  tr_files += append_to_list(AEOTD_wav,AEOTD_lab+"train_label_man.csv")
   # val_files += append_to_list(AEOTD_wav,AEOTD_lab+"val_label_man.csv")

    #DLFD = path + "DLFD\\"
    #tr_files += append_to_list(DLFD, DLFD + "new_train_label.csv")
    #val_files += append_to_list(DLFD, DLFD + "new_val_label.csv")

    #OAC_wav = "C:\\myProjects\\THESIS\\Orchive\\orchive-extract\\extract\\"
    #OAC_lab = path + "OAC\\"
   # tr_files += append_to_list(OAC_wav, OAC_lab + "new_train_label.csv")
    #val_files += append_to_list(OAC_wav, OAC_lab + "new_val_label.csv")

    #large_data = "C:\\myProjects\\THESIS\\DeepAL_ComParE\\DeepAL_ComParE\\ComParE2019_OrcaActivity\\wav\\"
    #large_data_csv = "C:\\myProjects\\THESIS\\DeepAL_ComParE\\DeepAL_ComParE\\ComParE2019_OrcaActivity\\lab\\DeepAL_ComParE.csv"
    #data_orig = pd.read_csv(large_data_csv)

  #  small_data = "C:\\myProjects\\THESIS\\Test_Pipeline01\\wav\\"
  #  small_data_csv = "C:\\myProjects\\THESIS\\Test_Pipeline01\\labels\\labels.csv"
  #  data_small = pd.read_csv(small_data_csv)

  #  train, test = np.split(data_orig.sample(frac=1), [int(.85 * len(data_orig))])
   # test.to_csv(TEST_FILES_CSV)

   # tr_files, val_files = train_test_split(sorted(wav_path + train.file_name.values), test_size=0.15, random_state=0)


    #X_train, X_test, y_train, y_test = train_test_split(
    #            filenames, labels, test_size = 0.2, random_state = 0)

    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=0)

    #X_test.to_csv(TEST_FILES_CSV)

    #print("X_train ", len(X_train))
    #print("X_val ", len(X_val))
    #print("X_test ", len(X_test))


    # In[]:

    def file_to_label(file_to_label,wav_path,csv_path):
        csv = pd.read_csv(csv_path)
        file_to_label.update({wav_path + k: v for k, v in zip(csv.file_name.values,csv.label.values)})
        return file_to_label

    file_to_label_train = {}
    file_to_label(file_to_label_train,path, path + "train_lab.csv")
    #file_to_label(file_to_label_train,AEOTD_wav,AEOTD_lab+"train_label_man.csv")
    #file_to_label(file_to_label_train,DLFD, DLFD + "new_train_label.csv")
    #file_to_label(file_to_label_train,OAC_wav, OAC_lab + "new_train_label.csv")

    file_to_label_val = {}
    file_to_label(file_to_label_val,path, path + "val_lab.csv")
   # file_to_label(file_to_label_val,AEOTD_wav, AEOTD_lab + "val_label_man.csv")
   # print("VAL LABEL 1", len(file_to_label_val))
   # file_to_label(file_to_label_val,DLFD, DLFD + "new_val_label.csv")
   # print("VAL LABEL 2", len(file_to_label_val))
   # file_to_label(file_to_label_val,OAC_wav, OAC_lab + "new_val_label.csv")
   # print("VAL LABEL 3", len(file_to_label_val))

    #file_to_label = {AEOTD_wav + k: v for k, v in zip(train.file_name.values,train.label.values)}

    list_labels = ["noise","orca"]  # Unique labels orca and noise
    print('list_labels ', list_labels)

    label_to_int = {k: v for v, k in enumerate(list_labels)}
    print('label_to_int ', label_to_int)

#    int_to_label = {v: k for k, v in label_to_int.items()}
#    print('int to label ', int_to_label)

    file_to_int_train = {k: label_to_int[v] for k, v in file_to_label_train.items()}
    file_to_int_val = {k: label_to_int[v] for k, v in file_to_label_val.items()}


    # In[]:
    def get_model_mel():

        inp = Input(shape=(128, 256, 1))
        norm_inp = BatchNormalization()(inp)
        img_1 = Convolution2D(16, kernel_size=(3, 3), activation=activations.relu)(norm_inp)
        img_1 = GlobalMaxPool2D()(img_1)
        img_1 = Dropout(rate=0.1)(img_1)
        dense_1 = Dense(1, activation=activations.softmax)(img_1)

        model = models.Model(inputs=inp, outputs=dense_1)

        opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile(optimizer=opt, loss="binary_crossentropy",
                      metrics=["acc"])

        model.summary()

        return model

    # In[15]:
    #model = get_model_mel()
    model = resnet.ResnetBuilder.build_resnet_18((128,256,1),1)
    opt = optimizers.Adam(lr=0.00001, beta_1=0.5, beta_2=0.999, amsgrad=False)
    model.compile(optimizer =opt,loss="binary_crossentropy", metrics=["acc"])
    model.summary()

 #   for layer in model.layers:
 #       print(layer.name, layer.input.shape, layer.output.shape)

  #  layer_dict = dict([(layer.name, layer) for layer in model.layers])
  #  print("layer dict ", layer_dict)
      # can be any integer from 0 to 511, as there are 512 filters in that layer

    # build a loss function that maximizes the activation
    # of the nth filter of the layer considered



    # In[]:
 #   dl = Dataloader(file_to_int_val)
 #   gradcsv= pd.read_csv(OAC_lab+"new_val_label.csv")
 #   T = []
 #   L = []
 #   for i in range(0,10):
 #       if gradcsv.label.values[i] == "noise":
 #           L.append(0)
 #       else:
 #           L.append(1)
 #       x = dl.load_audio_file(OAC_wav+gradcsv.file_name.values[i])
 #       T.append(x)
 #   train_files = np.array(T)[:,:, :, np.newaxis]

 #   my_debug_weights = MyDebugWeights(train_files, L, WEIGHTS_CSV, GRADIENTS_CSV)

    # In[]:
    training_generator = Train_Generator(tr_files, file_to_int_train,augment=True,batch_size=batch_size)
    validation_generator = Train_Generator(val_files, file_to_int_val,augment=False,batch_size=batch_size)

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5,mode='max',
                              patience=4, min_delta=1e-3)

    #tb = TensorBoard(log_dir='./Graph/{}'.format(ID), histogram_freq=0, write_graph=True, write_images=True)

    earlystop = EarlyStopping(monitor='val_acc', patience=10, mode='max')
    mcheckpoint= ModelCheckpoint(BEST_MODEL,monitor='val_acc',save_best_only=True,mode='max')

    start = timeit.default_timer()
    H = model.fit_generator(generator=training_generator, epochs=epochs, steps_per_epoch=len(tr_files) // batch_size,
                            validation_data=validation_generator, validation_steps=len(val_files) // batch_size,use_multiprocessing=False,
                            workers=multiprocessing.cpu_count(),verbose=1,
                            callbacks=[mcheckpoint,reduce_lr])
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
    N = epochs
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
