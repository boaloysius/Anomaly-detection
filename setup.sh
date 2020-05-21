# models
mkdir -p ../models
mkdir -p ../models/IR-MNIST
mkdir -p ../models/UCSDped1
mkdir -p ../models/UCSDped2

# data
mkdir -p ../data
cd ../data
# UCSD
wget http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz
tar -xvf UCSD_Anomaly_Dataset.tar.gz
rm UCSD_Anomaly_Dataset.tar.gz