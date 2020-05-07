# In[0]:
import pandas as pd
from pathlib import Path


from Train_Generator import *
from OrganizeData import *
import matplotlib.pyplot as plt
from keras import optimizers, losses, activations, models
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,  ModelCheckpoint
import keras.backend as K

#from DebugWeights import *
import timeit
import resnet
import multiprocessing
import argparse


ap = argparse.ArgumentParser()

ap.add_argument("--data-dir",type=str,help="path for training and val files")

ap.add_argument("--freq-compress", type=str,default='linear',help="define compression type")

ap.add_argument("-m", "--model", type=str, default=None,
	help="path to model checkpoint to load in case you want to resume a model training from where you left of")

ap.add_argument("--res-dir", type=str,default=None, help="path to model results plot,weights,checkpoints etc.")

ap.add_argument("-ie", "--initial-epoch", type=int, default=0,
	help="epoch to restart training at")

ap.add_argument("--batch", type=int,default=32, help="choose batch size")

ap.add_argument("--n-epochs", type=int, default=50,help="number of epochs")



ARGS = ap.parse_args()

    # In[2]:
if __name__=='__main__':

    # In[2]:

    dir = ARGS.res_dir

    HISTORY_CSV = "{}".format(dir) + "/history.csv"
    TEST_FILES_CSV = "{}".format(dir) + "/test_files.csv"
    MODEL_STRUCTURE_JSON = "{}".format(dir) + "/model_structure.json"
    WEIGHTS_CNN = "{}".format(dir) + "/weights.h5"
    BEST_MODEL = "{}".format(dir) + "/best_model.h5"
    PLOT = "{}".format(dir) + "/plot.png"
    RESULTS_CSV = "{}".format(dir) + "/results.csv"
    WEIGHTS_CSV = "{}".format(dir) + "/weights.csv"


    # In[]:

    data_dir = ARGS.data_dir
    tr_files, file_to_label_train = findcsv("train",data_dir)
    val_files, file_to_label_val = findcsv("val",data_dir)

    list_labels = ["noise","orca"]  # Unique labels orca and noise
    print('list_labels ', list_labels)

    label_to_int = {k: v for v, k in enumerate(list_labels)}
    print('label_to_int ', label_to_int)

    file_to_int_train = {k: label_to_int[v] for k, v in file_to_label_train.items()}
    file_to_int_val = {k: label_to_int[v] for k, v in file_to_label_val.items()}

    training_generator = Train_Generator(tr_files, file_to_int_train, freq_compress=ARGS.freq_compress, augment=True,
                                         batch_size=ARGS.batch)
    validation_generator = Train_Generator(val_files, file_to_int_val, freq_compress=ARGS.freq_compress, augment=False,
                                           batch_size=ARGS.batch)

    # In[15]:

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, mode='max',
                                  patience=4, min_delta=1e-3)

    earlystop = EarlyStopping(monitor='val_acc', patience=10, mode='max')

    if ARGS.model is None:
        check = BEST_MODEL
    else:
        check = ARGS.model
    mcheckpoint = ModelCheckpoint(check, monitor='val_acc', save_best_only=True, mode='max')

    if ARGS.model is None:
        print("[INFO] compiling model...")
        model = resnet.ResnetBuilder.build_resnet_18((128, 256, 1), 1)
        opt = optimizers.Adam(lr=0.00001, beta_1=0.5, beta_2=0.999, amsgrad=False)
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["acc"])
        model.summary()
    # otherwise, we're using a checkpoint model
    else:
        # load the checkpoint from disk
        print("[INFO] loading {}...".format(ARGS.model))
        model = models.load_model(ARGS.model)
        print("[INFO] current learning rate: {}".format(
            K.get_value(model.optimizer.lr)))


    H = model.fit_generator(generator=training_generator, epochs=ARGS.n_epochs, initial_epoch=ARGS.initial_epoch,
                            steps_per_epoch=len(tr_files) // ARGS.batch,
                            validation_data=validation_generator, validation_steps=len(val_files) // ARGS.batch,
                            use_multiprocessing=False,
                            workers=multiprocessing.cpu_count(), verbose=1,
                            callbacks=[mcheckpoint, reduce_lr])


    # In[]:

    model_structure = model.to_json()# convert the NN into JSON
    f = Path(MODEL_STRUCTURE_JSON) # write the JSON data into a text file
    f.write_text(model_structure) # Pass in the data that we want to write into the file.

    model.save_weights(WEIGHTS_CNN)
    pd.DataFrame.from_dict(H.history).to_csv(HISTORY_CSV,index=False)


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
