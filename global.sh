. ~/.bashrc
RAND=$RANDOM$RANDOM
gcloud ml-engine jobs submit training kidzbop$RAND --stream-logs \
       --runtime-version 1.4 --job-dir gs://fashiondataset/kidzbop$RAND \
       --package-path root/trainer --module-name trainer.train \
       --region europe-west1 --config root/trainer/config.yaml \
