from keras import backend as K
from keras.callbacks import Callback
import numpy as np
import datetime
from csv import DictWriter,writer

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

def append_dict_as_row(file_name, dict_of_elem, field_names):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        dict_writer = DictWriter(write_obj, fieldnames=field_names)
        # Add dictionary as wor in the csv
        dict_writer.writerow(dict_of_elem)

def calc_stats(W):
    return np.linalg.norm(W, 2),np.mean(W), np.std(W)

def calc_stats01(W):
    return np.linalg.norm(W), np.mean(W), np.std(W)

def get_gradients(inputs, labels, model):
    opt = model.optimizer
    loss = model.total_loss
    weights = [weight for weight in model.trainable_weights]
    grads = opt.get_gradients(loss, weights)
    grad_fn = K.function(inputs=[model.inputs[0],
                                 model.sample_weights[0],
                                 model.targets[0],
                                 K.learning_phase()],
                         outputs=grads)
    grad_values = grad_fn([inputs, np.ones(len(inputs)), labels, 1])
    return grad_values


class MyDebugWeights(Callback):

    def __init__(self,train_files, train_labels,weights_csv,gradients_csv):
        super(MyDebugWeights, self).__init__()
        self.train_files = train_files
        self.train_labels = train_labels
        self.weights = []
        self.tf_session = K.get_session()
        self.weights_csv = weights_csv
        self.gradients_csv = gradients_csv
        self.field_names = ['gradient','norm','mean','std']
        append_list_as_row(self.gradients_csv,self.field_names)
        self.field_names02=['epoch','layer','norm','mean','std']
        append_list_as_row(self.weights_csv,self.field_names02)


    def on_epoch_end(self, epoch, logs=None):

        gradients = get_gradients(self.train_files,self.train_labels,self.model)

        for i in range(len(gradients)):
            n, m, s = calc_stats01(gradients[i])
            row_dict = {'gradient': i, 'norm':n,'mean':m,'std':s}
            append_dict_as_row(self.gradients_csv, row_dict, self.field_names)
            #print("i ", i)
            #print("grad_{:d}".format(i), calc_stats01(gradients[i]))
        for layer in self.model.layers:
            name = layer.name
            for i, w in enumerate(layer.weights):
                w_value = w.eval(K.get_session())
                w_norm, w_mean, w_std = calc_stats(np.reshape(w_value, -1))
                self.weights.append((epoch, "{:s}/W_{:d}".format(name, i),
                                     w_norm, w_mean, w_std))

    def on_train_end(self, logs=None):
        for e, k, n, m, s in self.weights:

            row_dict = {'epoch':e,'layer': k, 'norm':n,'mean':m,'std':s}
            append_dict_as_row(self.weights_csv, row_dict, self.field_names02)
            #print("{:3d} {:20s} {:7.3f} {:7.3f} {:7.3f}".format(e, k, n, m, s))
