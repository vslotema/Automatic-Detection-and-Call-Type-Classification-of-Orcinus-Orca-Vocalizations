To run e.g. the pre-trained classifier 
e.g. Pipeline: python3 Pipeline.py --data-dir /mnt/c/myProjects/THESIS/csv/ORCASPOT_csv/ --res-dir /results/ --model /model.h5 --initial-epoch 7 --freq-compress linear

e.g. predict: python3 predict.py --data-dir /mnt/c/myProjects/THESIS/csv/ORCASPOT_csv/ --res-dir /results/ --freq-compress linear

Required arguments:

--data-dir
--res-dir -> need to create directory beforehand, not done by the script.

To generate augmented noise files using librosa
e.g. python3 generate_noise.py --data-dir /Users/charlottekruss/THESIS/DeepAL/noises/ --pitch-shift 1 --time-stretch 1 --add-noise 1
