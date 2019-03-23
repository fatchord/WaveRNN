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

# Samples

### Neural TTS output
"The buses aren't the problem, they actually provide a solution."
<audio src="https://github.com/fatchord/model_outputs/blob/master/TTS_1.wav?raw=true" controls preload></audio>
"The quick brown fox jumps over the lazy dog."
<audio src="https://github.com/fatchord/model_outputs/blob/master/TTS_2.wav?raw=true" controls preload></audio>
"Never going to give you up, never going to let you down, never going to turn around and desert you."
<audio src="https://github.com/fatchord/model_outputs/blob/master/TTS_3.wav?raw=true" controls preload></audio>
"George Washington was the first president of the United States."
<audio src="https://github.com/fatchord/model_outputs/blob/master/TTS_4.wav?raw=true" controls preload></audio>
"He reads books."
<audio src="https://github.com/fatchord/model_outputs/blob/master/TTS_6.wav?raw=true" controls preload></audio>
"He has read the whole thing."
<audio src="https://github.com/fatchord/model_outputs/blob/master/TTS_5.wav?raw=true" controls preload></audio>
"I want to see how this text to speech system can generalize, hopefully to sentences it has never seen before."
<audio src="https://github.com/fatchord/model_outputs/blob/master/TTS_7.wav?raw=true" controls preload></audio>
"In nineteen seventy two, a crack commando unit was sent to prison by a military court for a crime they didn't commit."
<audio src="https://github.com/fatchord/model_outputs/blob/master/TTS_8.wav?raw=true" controls preload></audio>
"These men promptly escaped from the maximum security stockage to the Los Angeles underground."
<audio src="https://github.com/fatchord/model_outputs/blob/master/TTS_9.wav?raw=true" controls preload></audio>
"Today, still wanted by the government, they survive as soldiers of fortune."
<audio src="https://github.com/fatchord/model_outputs/blob/master/TTS_10.wav?raw=true" controls preload></audio>
"If you have a problem, if no-one else can help, and if you can find them."
<audio src="https://github.com/fatchord/model_outputs/blob/master/TTS_11.wav?raw=true" controls preload></audio>
"Maybe you can hire, the A-Team."
<audio src="https://github.com/fatchord/model_outputs/blob/master/TTS_12.wav?raw=true" controls preload></audio>


## Vocoder output (Mel Spectrogram Input - Test Set)

| Generated | Ground Truth |
| ------------- | ------------- |
| <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_0_generated.wav?raw=true" controls preload></audio>  | <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_0_target.wav?raw=true" controls preload></audio>  |
| <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_1_generated.wav?raw=true" controls preload></audio>  | <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_1_target.wav?raw=true" controls preload></audio>  |
| <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_2_generated.wav?raw=true" controls preload></audio>  | <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_2_target.wav?raw=true" controls preload></audio>  |
| <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_3_generated.wav?raw=true" controls preload></audio>  | <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_3_target.wav?raw=true" controls preload></audio>  |
| <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_4_generated.wav?raw=true" controls preload></audio>  | <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_4_target.wav?raw=true" controls preload></audio>  |
| <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_5_generated.wav?raw=true" controls preload></audio>  | <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_5_target.wav?raw=true" controls preload></audio>  |
| <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_6_generated.wav?raw=true" controls preload></audio>  | <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_6_target.wav?raw=true" controls preload></audio>  |
| <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_7_generated.wav?raw=true" controls preload></audio>  | <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_7_target.wav?raw=true" controls preload></audio>  |
| <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_8_generated.wav?raw=true" controls preload></audio>  | <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_8_target.wav?raw=true" controls preload></audio>  |
| <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_9_generated.wav?raw=true" controls preload></audio>  | <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_9_target.wav?raw=true" controls preload></audio>  |
| <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_10_generated.wav?raw=true" controls preload></audio>  | <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_10_target.wav?raw=true" controls preload></audio>  |
| <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_11_generated.wav?raw=true" controls preload></audio>  | <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_11_target.wav?raw=true" controls preload></audio>  |
| <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_12_generated.wav?raw=true" controls preload></audio>  | <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_12_target.wav?raw=true" controls preload></audio>  |
| <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_13_generated.wav?raw=true" controls preload></audio>  | <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_13_target.wav?raw=true" controls preload></audio>  |
| <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_14_generated.wav?raw=true" controls preload></audio>  | <audio src="https://github.com/fatchord/model_outputs/blob/master/576k_steps_14_target.wav?raw=true" controls preload></audio>  

# Pretrained Models

Coming Soon

# References and Resources





