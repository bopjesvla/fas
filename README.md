# Fashion Classification Competition

*nerds trying to understand fashion*


Using keras to re-train a pre-trained model?

## Setup

To download all the training images, unpack all the ZIP files first (train, test, validation), and then run this:
```
py .\downloader.py .\train.json .\traindata\
py .\downloader.py .\test.json .\testdata\
py .\downloader.py .\validation.json .\validationdata\
```
(make sure the folders are the same, as this is in the gitignore)



## Useful links


..* [Pretrained Models Zoo (list of object detection pretrain models)](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

..& [Quickstart GCP ML with pretrained model](https://github.com/tensorflow/models/blob/676a4f70c20020ed41b533e0c331f115eeffe9a3/research/object_detection/g3doc/running_pets.md)

