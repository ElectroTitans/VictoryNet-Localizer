
sudo apt-get install zip -y
sudo apt-get install unzip -y
sudo apt-get -y install python3-pip
sudo pip3 install -r requirements.txt 

file="google-cloud-sdk-101.0.0-linux-x86_64.tar.gz"
link="https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/"

curl -L "$link""$file" | tar xz 
CLOUDSDK_CORE_DISABLE_PROMPTS=1 ./google-cloud-sdk/install.sh

gcloud auth activate-service-account --key-file gcpkey.json
mkdir Data/

gsutil cp gs://victorynet-trainingdata/current_dataset.zip dataset.zip
unzip dataset.zip -d ./

chmod +x ./train.sh
