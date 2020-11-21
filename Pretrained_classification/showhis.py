import pandas as pd

his = pd.read_csv('/Users/charlottekruss/THESIS/RUNS/res-endecoder-MAE-wo-batchnorm/history.csv')
ae = pd.read_csv('/Users/charlottekruss/THESIS/RUNS/res-autoencoder-OSD-mse-sigmoid-mel/history.csv')

print(his)

print(his['val_loss'].mean())
print('val mse  min - mean ', his['val_mae'].min(), his['val_mae'].mean())
print('val loss min - mean ',his['val_loss'].min(), his['val_loss'].mean())

print(ae)
print(ae['val_loss'].mean())
print('val mse  min - mean ', ae['val_mean_squared_error'].min(), ae['val_mean_squared_error'].mean())
print('val loss min - mean ',ae['val_loss'].min(), ae['val_loss'].mean())

