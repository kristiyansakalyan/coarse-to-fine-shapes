#!/bin/bash

# echo "Install cudatoolkit"
# conda install -y -c anaconda cudatoolkit

echo "Installing dependencies for pip"
pip install torch==1.4.0
pip install torchvision==0.5.0
pip install matplotlib==2.2.5
pip install tqdm==4.32.1
pip install open3d==0.9.0
pip install trimesh==3.7.12
pip install scipy==1.5.1    

echo "Installing local cuda emd dependency"
cd metrics/PyTorchEMD
python setup.py install
cd ../..