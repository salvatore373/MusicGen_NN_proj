# Salvatore Michele Rago, Mattia Maffongelli: MusicGen_NN
In this GitHub repository, we present the study and the corresponding implemented code of the paper titled ["MusicGen, a simple and controllable model for music generation"](https://arxiv.org/abs/2306.05284).

MusicGen is an **auto-regressive Transformer-based decoder** trained using a 32kHz EnCodec tokenizer (However, due to a lack of resources, our model is trained using a 24kHz EnCodec tokenizer) with 4 codebooks sampled at 50 Hz.
The purpose is to create a **versatile framework** for modeling multiple parallel streams of acoustic tokens, enabling the generation of music aligned with specified harmonic and melodic structures, thanks to the **text** and **melody conditioning**. 

The model is based on **EnCodec**, an audio tokenizer that uses quantized units to reconstruct audio with high fidelity, **RVQ**(Residual Vector Quantization), a compression model utilized to efficiently represent and store audio information, which introduces a residual component, capturing the difference between the original signal and the quantized representation and, finally, on particular **codebook interleaving patterns**. These one refer to a specific ways of organizing and using the codebooks during the generation or processing of audio: in our case, we used the **"delay pattern"**, which at each time step "s", it considers codebooks with an increasing delay. Subsequently, for each codebook "k", it looks back at previous time steps, incorporating a delay of "k - 1" steps. The aim is to capture temporal dependencies in the audio data, considering how the audio evolves over time and incorporating information from past steps.

In text conditioning, we employed the pre-trained **T5-Encoder**. This encoder takes a textual description that matches the input audio X and computes a conditioning tensor with dimensions T_C = length of the sequence * D, where D is the inner dimension used in the autoregressive model.

For melody conditioning, we utilized the **Librosa** library to handle audio manipulation. Specifically, we extracted the chromagram from the input audio and then applied filters to adapt the dimensions, resulting in the final tensor.

All the specific implementations are described in the notebook.

[Google Colab Notebook](https://colab.research.google.com/drive/1I6lvaCnEkBErb19yE64xF93MJeZnO5OM?usp=sharing)

## Installation

All the codes in this repository require Python 3.9 or above, PyTorch 2.1.0 or above, and other libraries. 

To ensure everything is executed correctly, please run the following commands:

```bash
# Don't run this if you already have PyTorch installed.
python -m pip install 'torch==2.1.0'
pip install torchaudio
# ffmpeg, required for other libraries
sudo apt-get install ffmpeg
# librosa
pip install librosa
# einops
pip install einops
# datasets
pip install datasets
# soundfile
pip install soundfile
# transformers library, by HuggingFace
pip install git+https://github.com/huggingface/transformers.git
# numpy
pip install numpy
```

# Learn more
Learn more about MusicGen visiting the official [GitHub Repository](https://github.com/facebookresearch/audiocraft/tree/main).
