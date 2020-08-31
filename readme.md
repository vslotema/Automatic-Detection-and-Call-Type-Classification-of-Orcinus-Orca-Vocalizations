
Consists of two classifiers, one that handles binary classification and another one multi, both are based on a ResNet18 architecture.

To run:
e.g. main:
python3 main.py --data-dir /mnt/c/myProjects/THESIS/csv/ORCASPOT_csv/ --res-dir /results/ --model /model.h5 --initial-epoch 7 --freq-compress linear  

e.g. predict:
python3 predict.py --data-dir /mnt/c/myProjects/THESIS/csv/ORCASPOT_csv/ --res-dir /results/ --freq-compress linear

Required arguments:
- --data-dir 
- --res-dir -> need to create directory beforehand, not done by the script. 