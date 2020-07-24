# In[0]:

from pathlib import Path
from Train_Generator import *
from OrganizeData import *
import Resnet18
import matplotlib.pyplot as plt


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
import argparse
import math
import numpy as np
import pandas as pd
import statistics

import re

ap = argparse.ArgumentParser()

ap.add_argument("--n-threads", type=int, default=4, help="number of working threads")

ap.add_argument("--data-dir", type=str,
                help="path for training and val files")

ap.add_argument("--freq-compress", type=str, default='linear', help="define compression type")

ap.add_argument("-m", "--model", type=str, default=None,
                help="path to model checkpoint to load in case you want to resume a model training from where you left of")

ap.add_argument("--res-dir", type=str, help="path to model results plot,weights,checkpoints etc.")

ap.add_argument("-pm","--pretrained_mod",default=None,type=str,help="path for results of this classifier")

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

def getModel(pm,lr,n_output):

    if pm is None:
        model = Resnet18.ResnetBuilder.build_resnet_18((128, 256, 1), n_output)
    else:
        print("[INFO] compiling model...")
        f = Path(pm + "model_structure.json")
        model_structure = f.read_text()

        # Recreate the Keras model object from the json data
        ae = model_from_json(model_structure)

        initial_weights = {}
        for layer in ae.get_layer('encoder').layers[-16:]:
            if not re.findall("add|activation",layer.name):
                print(np.shape(layer.get_weights()))
                print("layer name: ", layer.name)
                initial_weights.update({layer.name:layer.get_weights()})

        print(initial_weights.keys())

        # Re-load the model's trained weights
        ae.load_weights(ARGS.pretrained_mod + "best_model.h5")
        # Load last residual layer with initial weights
        for layer in ae.get_layer('encoder').layers[-16:]:
            if not re.findall("add|activation",layer.name):
                print("layer name: ", layer.name)
                layer.set_weights(initial_weights.get(layer.name))

        y = ae.get_layer('encoder').get_output_at(-1)

        block_shape = K.int_shape(y)
        pool2 = AveragePooling2D(pool_size=(block_shape[1], block_shape[2]),
                                 strides=(1, 1))(y)

        flatten1 = Flatten()(pool2)
        dense = Dense(units=n_output,
                      activation="softmax",use_bias=True,kernel_initializer="he_normal")(flatten1)

        model = Model(inputs=ae.input, outputs=dense)

    opt = optimizers.Adam(lr=lr, beta_1=0.5, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["acc"])
    model.summary()
    return model
    # otherwise, we're using a checkpoint model

# In[2]:
if __name__ == '__main__':

    # In[2]:

    # In[]:
    print("{} compression".format(ARGS.freq_compress))
    data_dir = ARGS.data_dir
    tr_files, file_to_label_train = findcsv("train", data_dir)
    val_files, file_to_label_val = findcsv("val", data_dir)
    test_files, file_to_label_test = findcsv("test",data_dir)

    list_labels = getUniqueLabels(data_dir)
    print("Unique Values: ",list_labels)

    n_output = len(list_labels)
    print("n output ", n_output)

    label_to_int = {k: v for v, k in enumerate(list_labels)}

    file_to_int_train = {k: label_to_int[v] for k, v in file_to_label_train.items()}
    file_to_int_val = {k: label_to_int[v] for k, v in file_to_label_val.items()}
    file_to_int_test = {k: label_to_int[v] for k, v in file_to_label_test.items()}

    # In[15]:
    ARGS.lr *= ARGS.batch
    print("learning rate ", ARGS.lr)


    if ARGS.model is None:
        model = getModel(ARGS.pretrained_mod,ARGS.lr,n_output)

    else:
        # load the checkpoint from disk
        print("[INFO] loading {}...".format(ARGS.model))
        model = models.load_model(ARGS.model)
        print("[INFO] current learning rate: {}".format(
            K.get_value(model.optimizer.lr)))

    patience_lr = math.ceil(ARGS.lr_patience_epochs / ARGS.epochs_per_eval)
    patience_lr = int(max(1, patience_lr))
    print("patience lr ", patience_lr)

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, mode='max',
                                  patience=patience_lr, min_delta=1e-3, cooldown=1, verbose=1)

    earlystop = EarlyStopping(monitor='val_acc', patience=ARGS.early_stopping_patience_epochs, mode='max')

    training_generator = Train_Generator("train", tr_files, file_to_int_train, freq_compress=ARGS.freq_compress,
                                         augment=True,
                                         batch_size=ARGS.batch, add_noise=ARGS.add_noise)
    validation_generator = Train_Generator("val", val_files, file_to_int_val, freq_compress=ARGS.freq_compress,
                                           augment=False,
                                           batch_size=ARGS.batch)

    all_scores = []
    for i in range(10):
        dir = ARGS.res_dir

        HISTORY_CSV = "{}".format(dir) + "{}_history.csv".format(i)
        TEST_FILES_CSV = "{}".format(dir) + "{}_test_files.csv".format(i)
        MODEL_STRUCTURE_JSON = "{}".format(dir) + "{}_model_structure.json".format(i)
        WEIGHTS_CNN = "{}".format(dir) + "{}_weights.h5".format(i)
        BEST_MODEL = "{}".format(dir) + "{}_best_model.h5".format(i)
        PLOT = "{}".format(dir) + "{}_plot.png".format(i)
        RESULTS_CSV = "{}".format(dir) + "{}_res_test.csv".format(i)
        WEIGHTS_CSV = "{}".format(dir) + "{}_weights.csv".format(i)
        LOG = "{}".format(dir) + "{}_log/".format(i)
        SCORES_CSV = "{}".format(dir) + "{}_scores.csv".format(i)

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
                                callbacks=[mcheckpoint, reduce_lr, earlystop,logger])

        # In[]:

        model_structure = model.to_json()  # convert the NN into JSON
        f = Path(MODEL_STRUCTURE_JSON)  # write the JSON data into a text file
        f.write_text(model_structure)  # Pass in the data that we want to write into the file.

        model.save_weights(BEST_MODEL)
        #pd.DataFrame.from_dict(H.history).to_csv(HISTORY_CSV, index=False)

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

    acc_scores = open(ARGS.res_dir + "acc_scores.csv","w+")
    print("All SCORES {}".format(all_scores))
    print("MAX SCORE:{} , FOLD:{}".format(max(all_scores),all_scores.index(max(all_scores))))
    print("AVERAGE SCORE {}".format(statistics.mean(all_scores)))
    acc_scores.write("All SCORES {}".format(all_scores))
    acc_scores.write("MAX SCORE:{} , FOLD:{}".format(max(all_scores),all_scores.index(max(all_scores))))
    acc_scores.write("AVERAGE SCORE {}".format(statistics.mean(all_scores)))
    acc_scores.close()