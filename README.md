# WaveRNN

Pytorch implementation of Deepmind's WaveRNN model from [Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435v1)

# Dependencies

* Python 3.x
* Pytorch 1.0.1 with CUDA
* Numpy 1.16.0
* Librosa 0.6.0

# Training

Edit **hparams.py** and point **wav_path** to your dataset 

> python preprocess.py

Once that's completed you can start training:

> python train.py



