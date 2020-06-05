# In[0]:

from pathlib import Path
from Train_Generator import *
from OrganizeData import *
import matplotlib.pyplot as plt
from keras import optimizers, losses, activations, models
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import keras.backend as K
import tensorflow as tf
import Resnet18
import argparse
import math

ap = argparse.ArgumentParser()

ap.add_argument("--n-threads", type=int, default=4, help="number of working threads")

ap.add_argument("--data-dir", type=str,
                help="path for training and val files")

ap.add_argument("--freq-compress", type=str, default='linear', help="define compression type")

ap.add_argument("-m", "--model", type=str, default=None,
                help="path to model checkpoint to load in case you want to resume a model training from where you left of")

ap.add_argument("--res-dir", type=str, help="path to model results plot,weights,checkpoints etc.")

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
    default=2,
    help="The number of batches to run in between evaluations.",
)

ARGS = ap.parse_args()

# In[2]:
if __name__ == '__main__':

    # In[2]:

    dir = ARGS.res_dir

    HISTORY_CSV = "{}".format(dir) + "history.csv"
    TEST_FILES_CSV = "{}".format(dir) + "test_files.csv"
    MODEL_STRUCTURE_JSON = "{}".format(dir) + "model_structure.json"
    WEIGHTS_CNN = "{}".format(dir) + "weights.h5"
    BEST_MODEL = "{}".format(dir) + "best_model.h5"
    PLOT = "{}".format(dir) + "plot.png"
    RESULTS_CSV = "{}".format(dir) + "results.csv"
    WEIGHTS_CSV = "{}".format(dir) + "weights.csv"
    LOG = "{}".format(dir) + "log/"

    # In[]:
    print("{} compression".format(ARGS.freq_compress))
    data_dir = ARGS.data_dir
    tr_files, file_to_label_train = findcsv("train", data_dir)
    print("len train", len(tr_files))
    val_files, file_to_label_val = findcsv("val", data_dir)
    print("len val ", len(val_files))

    list_labels = getUniqueLabels(data_dir)
    print("Unique Values: ",list_labels)

    n_output = len(list_labels)
    print("n output ", n_output)

    label_to_int = {k: v for v, k in enumerate(list_labels)}

    file_to_int_train = {k: label_to_int[v] for k, v in file_to_label_train.items()}
    file_to_int_val = {k: label_to_int[v] for k, v in file_to_label_val.items()}

    training_generator = Train_Generator("train", tr_files, file_to_int_train, freq_compress=ARGS.freq_compress,
                                         augment=True,
                                         batch_size=ARGS.batch)
    validation_generator = Train_Generator("val", val_files, file_to_int_val, freq_compress=ARGS.freq_compress,
                                           augment=False,
                                           batch_size=ARGS.batch)

    # In[15]:
    ARGS.lr *= ARGS.batch
    print("learning rate ", ARGS.lr)

    patience_lr = math.ceil(ARGS.lr_patience_epochs / ARGS.epochs_per_eval)
    patience_lr = int(max(1, patience_lr))
    print("patience lr ", patience_lr)

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, mode='max',
                                  patience=patience_lr, min_delta=1e-3, cooldown=1, verbose=1)

    earlystop = EarlyStopping(monitor='val_acc', patience=ARGS.early_stopping_patience_epochs, mode='max')
    #tensorboard = keras.callbacks.TensorBoard(log_dir=LOG, histogram_freq=1)

    if ARGS.model is None:
        check = BEST_MODEL
    else:
        check = ARGS.model

    mcheckpoint = ModelCheckpoint(check, monitor='val_acc', save_best_only=True, mode='max')
    logger = keras.callbacks.CSVLogger(HISTORY_CSV,separator=",",append=True)

    if ARGS.model is None:
        print("[INFO] compiling model...")
        model = Resnet18.ResnetBuilder.build_resnet_18((128, 256, 1), n_output)
        opt = optimizers.Adam(lr=ARGS.lr, beta_1=0.5, beta_2=0.999, amsgrad=False)
        model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["acc"])
        model.summary()
    # otherwise, we're using a checkpoint model
    else:
        # load the checkpoint from disk
        print("[INFO] loading {}...".format(ARGS.model))
        model = models.load_model(ARGS.model)
        print("[INFO] current learning rate: {}".format(
            K.get_value(model.optimizer.lr)))

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
