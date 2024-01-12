# Salvatore Michele Rago, Mattia Maffongelli: MusicGen_NN
In this GitHub repository, we present the study and the corresponding implemented code of the paper titled ["MusicGen, a simple and controllable model for music generation"](https://arxiv.org/abs/2306.05284).

MusicGen is an **auto-regressive Transformer** model trained using a 32kHz EnCodec tokenizer (However, due to a lack of resources, our model is trained using a 24kHz EnCodec tokenizer) with 4 codebooks sampled at 50 Hz.
The purpose is to create a versatile framework for modeling multiple parallel streams of acoustic tokens, enabling the generation of music aligned with specified harmonic and melodic structures, thanks to the text and melody conditioning. 

