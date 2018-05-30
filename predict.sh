. ~/.bashrc
# gsutil cp gs://fashiondataset/model.h5 data/model.h5
RAND=$RANDOM$RANDOM
gcloud ml-engine jobs submit training kidzbop$RAND --stream-logs \
       --runtime-version 1.4 --job-dir gs://fashiondataset/kidzbop$RAND \
       --package-path root/trainer --module-name trainer.predict \
       --region europe-west1 --config root/trainer/config.yaml \
