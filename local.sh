gcloud ml-engine local train --module-name trainer.train \
       --package-path root/trainer -- \
       --job-dir ./tmp/test_script_gcp
