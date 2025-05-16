sudo apt update
sudo apt install python3-pip python3-venv git wget
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
wget https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Jester/20bnjester-v1-00
wget https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Jester/20bnjester-v1-01
wget https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Jester/20bnjester-v1-02
cat 20bn-jester-v1-?? | tar zx
mkdir -p data/jester
mkdir -P data/train
mkdir -p data/val
mkdir -p data/test
mv 20bnjester-v1-000/data/jester/frames/train data/train
mv 20bnjester-v1-000/data/jester/frames/val data/val
mv 20bnjester-v1-000/data/jester/frames/test data/test
