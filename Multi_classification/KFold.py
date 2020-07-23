# In[0]:

from pathlib import Path
from Train_Generator import *
from OrganizeData import *
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
import keras
from keras import Model
from keras import optimizers, losses, activations, models
from keras.models import model_from_json
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import keras.backend as K
from keras.layers import (
    Input,
    Dense,
    Flatten,
    AveragePooling2D
)

from keras.initializers import he_uniform
import random
import argparse
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,train_test_split

import re
from tqdm import tqdm
import statistics

ap = argparse.ArgumentParser()

ap.add_argument("--n-threads", type=int, default=4, help="number of working threads")

ap.add_argument("--data-dir", type=str,
                help="path for training and val files")

ap.add_argument("--freq-compress", type=str, default='linear', help="define compression type")

ap.add_argument("-m", "--model", type=str, default=None,
                help="path to model checkpoint to load in case you want to resume a model training from where you left of")

ap.add_argument("--res-dir", type=str, help="path to model results plot,weights,checkpoints etc.")

ap.add_argument("-pm", "--pretrained_mod", type=str, help="path for results of this classifier")

ap.add_argument("-ie", "--initial-epoch", type=int, default=0,
                help="epoch to restart training at")

ap.add_argument("--batch", type=int, default=32, help="choose batch size")

ap.add_argument("--n-epochs", type=int, default=100, help="number of epochs")

ap.add_argument(
    "--lr",
    "--learning_rate",
    type=float,
    default=1e-5,
    help="Initial learning rate. Will get multiplied by the batch size.",
)

ap.add_argument(
    "--beta1", type=float, default=0.5, help="beta1 for the adam optimizer."
)

ap.add_argument(
    "--lr_patience_epochs",
    type=int,
    default=8,
    help="Decay the learning rate after N/epochs_per_eval epochs without any improvements on the validation set.",
)

ap.add_argument(
    "--lr_decay_factor",
    type=float,
    default=0.5,
    help="Decay factor to apply to the learning rate.",
)

ap.add_argument(
    "--early_stopping_patience_epochs",
    metavar="N",
    type=int,
    default=20,
    help="Early stopping (stop training) after N/epochs_per_eval epochs without any improvements on the validation set.",
)

ap.add_argument(
    "--epochs_per_eval",
    type=int,
    default=1,
    help="The number of batches to run in between evaluations.",
)

ap.add_argument(
    "--add_noise",
    type=bool,
    nargs='?',
    const=True,
    default=False,
    help="if True, this will add noise to the training samples"
)

ARGS = ap.parse_args()


def writeFilestoCSV(ftl, csv):
    print("csv ", csv)
    csv.write("file_name,label\n")
    for k, v in ftl.items():
        split = k.split("/")
        csv.write(split[len(split)-1] + "," + v + "\n")
    csv.close()

def ftl(files, file_to_label):
    ftl = {}
    for f in files:
        ftl.update({f: file_to_label.get(f)})
    return ftl

def getModel(n_output):
    if ARGS.model is None:
        print("[INFO] compiling model...")
        f = Path(ARGS.pretrained_mod + "model_structure.json")
        model_structure = f.read_text()

        # Recreate the Keras model object from the json data
        ae = model_from_json(model_structure)

        # Re-load the model's trained weights
        print("LOOOP")
        initial_weights = {}
        for layer in ae.get_layer('encoder').layers[-16:]:
            if not re.findall("add|activation", layer.name):
                print(np.shape(layer.get_weights()))
                print("layer name: ", layer.name)
                initial_weights.update({layer.name: layer.get_weights()})

        print(initial_weights.keys())

        # print("initial weights ", initial_weights)
        ae.load_weights(ARGS.pretrained_mod + "best_model.h5")

        for layer in ae.get_layer('encoder').layers[-16:]:
            if not re.findall("add|activation", layer.name):
                print("layer name: ", layer.name)
                layer.set_weights(initial_weights.get(layer.name))

        # new_w = np.random.uniform(low=0.0,high=0.1,size=np.shape(last_layer.get_weights()))
        # new_weights = he_uniform(seed=random.randint(0,1000))(np.shape(last_layer.get_weights()))
        # print("array ", new_weights.numpy())

        # print("*************\n weights after ", last_layer.get_weights())
        y = ae.get_layer('encoder').get_output_at(-1)

        block_shape = K.int_shape(y)
        pool2 = AveragePooling2D(pool_size=(block_shape[1], block_shape[2]),
                                 strides=(1, 1))(y)

        flatten1 = Flatten()(pool2)
        dense = Dense(units=n_output,
                      activation="softmax", use_bias=True, kernel_initializer="he_normal")(flatten1)

        model = Model(inputs=ae.input, outputs=dense)


        opt = optimizers.Adam(lr=lr, beta_1=0.5, beta_2=0.999, amsgrad=False)
        model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["acc"])
        model.summary()
        # otherwise, we're using a checkpoint model
    else:
        # load the checkpoint from disk
        print("[INFO] loading {}...".format(ARGS.model))
        model = models.load_model(ARGS.model)
        print("[INFO] current learning rate: {}".format(
            K.get_value(model.optimizer.lr)))
    return model


# In[2]:
if __name__ == '__main__':

    # In[2]:

    # In[]:
    print("{} compression".format(ARGS.freq_compress))

    # In[15]:
    lr = ARGS.batch * ARGS.lr
    print("learning rate ", lr)



    data_dir = ARGS.data_dir
    files, file_to_label = findcsv("data", data_dir)

    list_labels = getUniqueLabels(data_dir)
    print("Unique Values: ", list_labels)

    n_output = len(list_labels)
    print("n output ", n_output)
    model = getModel(n_output)

    label_to_int = {k: v for v, k in enumerate(list_labels)}

    kf = KFold(n_splits=10)
    fold = 1
    all_scores = []
    for train_i, test_i in kf.split(files):
        #print("train i ", train_i)
        #print("test i ", test_i)
        dir = ARGS.res_dir

        TRAIN_CSV = open("{}".format(dir) + "{}_train.csv".format(str(fold)),"w+")
        VAL_CSV = open("{}".format(dir) + "{}_val.csv".format(str(fold)),"w+")
        TEST_CSV = open("{}".format(dir) + "{}_test.csv".format(str(fold)),"w+")
        HISTORY_CSV = "{}".format(dir) + "{}_history.csv".format(str(fold))
        TEST_FILES_CSV = "{}".format(dir) + "{}_test_files.csv".format(str(fold))
        MODEL_STRUCTURE_JSON = "{}".format(dir) + "{}_model_structure.json".format(str(fold))
        WEIGHTS_CNN = "{}".format(dir) + "{}_weights.h5".format(str(fold))
        BEST_MODEL = "{}".format(dir) + "{}_best_model.h5".format(str(fold))
        PLOT = "{}".format(dir) + "{}_plot.png".format(str(fold))
        RESULTS_CSV = "{}".format(dir) + "{}_res_test.csv".format(str(fold))
        SCORES_CSV = "{}".format(dir) + "{}_scores.csv".format(str(fold))
        LOG = "{}".format(dir) + "{}_log/".format(str(fold))
        fold += 1

        train_i,val_i = train_test_split(train_i,test_size=0.186)

        tr_files = [files[i] for i in train_i]
        file_to_label_train = ftl(tr_files,file_to_label)
        file_to_int_train = {k: label_to_int[v] for k, v in file_to_label_train.items()}
        writeFilestoCSV(file_to_label_train,TRAIN_CSV)

        val_files = [files[i] for i in val_i]
        file_to_label_val = ftl(val_files,file_to_label)
        file_to_int_val = {k: label_to_int[v] for k, v in file_to_label_val.items()}
        writeFilestoCSV(file_to_label_val, VAL_CSV)

        test_files = [files[i] for i in test_i]
        file_to_label_test = ftl(test_files, file_to_label)
        file_to_int_test = {k: label_to_int[v] for k, v in file_to_label_test.items()}
        writeFilestoCSV(file_to_label_test, TEST_CSV)

        training_generator = Train_Generator("train", tr_files, file_to_int_train, freq_compress=ARGS.freq_compress,
                                             augment=True,
                                             batch_size=ARGS.batch, add_noise=ARGS.add_noise)
        validation_generator = Train_Generator("val", val_files, file_to_int_val, freq_compress=ARGS.freq_compress,
                                               augment=False,
                                               batch_size=ARGS.batch)

        patience_lr = math.ceil(ARGS.lr_patience_epochs / ARGS.epochs_per_eval)
        patience_lr = int(max(1, patience_lr))
        print("patience lr ", patience_lr)

        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, mode='max',
                                      patience=patience_lr, min_delta=1e-3, cooldown=1, verbose=1)

        earlystop = EarlyStopping(monitor='val_acc', patience=ARGS.early_stopping_patience_epochs, mode='max')
        # tensorboard = keras.callbacks.TensorBoard(log_dir=LOG, histogram_freq=1)

        if ARGS.model is None:
            check = BEST_MODEL
        else:
            check = ARGS.model

        mcheckpoint = ModelCheckpoint(check, monitor='val_acc', save_best_only=True, mode='max')
        logger = keras.callbacks.CSVLogger(HISTORY_CSV, separator=",", append=True)

        H = model.fit_generator(generator=training_generator, epochs=ARGS.n_epochs, initial_epoch=ARGS.initial_epoch,
                                steps_per_epoch=math.floor(len(tr_files) // ARGS.batch),
                                validation_data=validation_generator,
                                validation_steps=math.ceil(len(val_files) // ARGS.batch),
                                use_multiprocessing=False,
                                workers=ARGS.n_threads, verbose=1,
                                callbacks=[mcheckpoint, reduce_lr, earlystop, logger])

        # In[]:


        dl = Dataloader(False, freq_compress=ARGS.freq_compress)


        batch_data = [dl.load_audio_file(fpath) for fpath in test_files]
        batch_data = np.array(batch_data)[:, :, :, np.newaxis]
        batch_labels = [file_to_int_test[fpath] for fpath in test_files]
        batch_labels = np.array(batch_labels)
        loss,acc = model.evaluate(batch_data,batch_labels)
        preds = model.predict(batch_data).tolist()

        print("score ", acc)
        all_scores.append(acc)

        classes_to_int = np.arange(len(list_labels))
        dict_scores = {key: [] for key in classes_to_int}

        def add_to_dict(scores):
            for i in range(len(scores)):
                dict_scores.get(i).append(scores[i])


        pred_1 = []
        pred_1_score = []
        pred_2 = []
        pred_2_score = []

        for i in preds:
            line = add_to_dict(i)
            idxs = np.argsort(i)[::-1][:2]  # get top2 indexes
            pred_1.append(list_labels[idxs[0]])
            pred_1_score.append(i[idxs[0]])
            pred_2.append(list_labels[idxs[1]])
            pred_2_score.append(i[idxs[1]])

        df_scores = pd.DataFrame(dict_scores)
        df_scores.to_csv(SCORES_CSV, index=False)

        file_name = []
        labels = []

        for i in test_files:
            file_name.append(i)
            labels.append(file_to_label_test[i])

        df = pd.DataFrame(file_name, columns=["file_name"])
        print("df len ", len(df))
        df['label'] = labels
        print("pred 1 len ", len(pred_1))
        df['pred_1'] = pred_1

        df['pred_1_score'] = pred_1_score
        df['pred_2'] = pred_2
        df['pred_2_score'] = pred_2_score

        df.to_csv(RESULTS_CSV, index=False)

        model_structure = model.to_json()  # convert the NN into JSON
        f = Path(MODEL_STRUCTURE_JSON)  # write the JSON data into a text file
        f.write_text(model_structure)  # Pass in the data that we want to write into the file.

        model.save_weights(BEST_MODEL)
        # pd.DataFrame.from_dict(H.history).to_csv(HISTORY_CSV, index=False)

        # In[19]:

        # plot the training loss and accuracy
        N = len(H.history['loss'])
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

    print("All SCORES ", all_scores)
    print("AVERAGE SCORE ", statistics.mean(all_scores))