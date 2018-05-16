sudo rm -r Data/
mkdir Data/

gsutil cp gs://victorynet-trainingdata/current_dataset.zip dataset.zip
unzip dataset.zip -d ./