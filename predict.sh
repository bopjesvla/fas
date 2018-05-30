. ~/.bashrc
gsutil cp gs://fashiondataset/model.h5 data/model.h5
python root/trainer/local_predict.py
