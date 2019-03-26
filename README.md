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
| Generated | Ground Truth |
| ------------- | ------------- |
| <audio src="https://github.com/fatchord/WaveRNN/blob/master/assets/436k_steps_target.wav?raw=true" controls preload></audio>  | <audio src="https://github.com/fatchord/WaveRNN/blob/master/assets/436k_steps_gen.wav?raw=true" controls preload></audio>  |
[Can be found here.](https://fatchord.github.io/model_outputs/)

# Pretrained Models

[Available Here](https://ufile.io/wyy4s)

# Acknowledgments

* [Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435v1)
* [keithito tacotron](https://github.com/keithito/tacotron)
* Special thanks to github users [G-Wang](https://github.com/G-Wang), [geneing](https://github.com/geneing)




