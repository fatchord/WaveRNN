# WaveRNN

Pytorch implementation of Deepmind's WaveRNN model from [Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435v1)

# Dependencies

* Python >= 3.6
* Pytorch 1.0.1 with CUDA
* Numpy 1.16.0
* Librosa 0.6.0


# Training

Edit **hparams.py**, point **wav_path** to your dataset and run: 

> python preprocess.py

Once that's completed you can start training:

> python train.py

# Samples

[Can be found here.](https://fatchord.github.io/model_outputs/)

# Pretrained Models

Currently there are two pretrained models available in the /pretrained/ folder':

* Single-speaker (LJ Speech) - 9bit mulaw - trained to ~400k steps
* Multi-speaker (Librispeech) - 9bit mulaw - trained to ~900k steps

# Acknowledgments

* [Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435v1)
* [keithito tacotron](https://github.com/keithito/tacotron)
* Special thanks to github users [G-Wang](https://github.com/G-Wang), [geneing](https://github.com/geneing)




