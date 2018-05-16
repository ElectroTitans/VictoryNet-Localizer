#! bash
sudo apt-get install zip -y
sudo apt-get install unzip -y
sudo apt-get -y install python3-pip
sudo pip3 install .

curl https://sdk.cloud.google.com | bash --disable-prompts
gcloud init --disable-prompts

gcloud auth activate-service-account --key-file gcpkey.json
mkdir Data/

gsutil cp gs://victorynet-trainingdata/current_dataset.zip dataset.zip
unzip dataset.zip -d Data/

