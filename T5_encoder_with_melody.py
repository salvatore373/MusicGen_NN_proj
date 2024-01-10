import torch
import torchaudio
from transformers import T5Tokenizer, T5ForConditionalGeneration
import librosa
import numpy as np
import soundfile as sf
import torch.nn.functional as F


class TextMelodyToTokenConverter:
    def __init__(self):
        self.text_linear_layer = torch.nn.Linear(768, 4)  # Assuming the original dimension is 768 for text

        for param in self.text_linear_layer.parameters():
            param.requires_grad_(False)

    def extract_dominant_bin(self, chromagram):
        # Assuming chromagram is a 2D tensor with shape (number of bins, number of frames)
        dominant_bin = torch.argmax(chromagram, dim=0)  # Select the dominant bin for each frame
        return dominant_bin

    # original code (simplified)
    def ver2(self, audio_times, sr):
        audio_times = audio_times.to(torch.float)
        self.nfft = 128
        self.spec = torchaudio.transforms.Spectrogram(n_fft=self.nfft, power=2, center=True,
                                                      pad=0, normalized=True)
        from librosa import filters
        self.fbanks = torch.from_numpy(filters.chroma(sr=sr, n_fft=self.nfft))

        spec = self.spec(torch.transpose(audio_times, 0, 1)).squeeze(1)
        raw_chroma = torch.einsum('cf,...ft->...ct', self.fbanks, spec)
        norm_chroma = torch.nn.functional.normalize(raw_chroma, dim=-2, eps=1e-6)
        from einops import rearrange
        norm_chroma = rearrange(norm_chroma, 'b d t -> b t d')

        # if True:
        if False:
            idx = norm_chroma.argmax(-1, keepdim=True)
            norm_chroma[:] = 0
            norm_chroma.scatter_(dim=-1, index=idx, value=1)

        return norm_chroma

    def convert_text_melody_to_tokens(self, text, audio_file_path):
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        model = T5ForConditionalGeneration.from_pretrained("t5-base")

        # ENCODER PART WITH T5-MODEL FOR TEXT
        text_input_ids = tokenizer(text, return_tensors="pt").input_ids

        with torch.no_grad():
            text_encoder_output = model.encoder(text_input_ids).last_hidden_state

        # Generate the conditioning tensor for text, having dimension T_C * D
        # where T_C = length of the sequence and D = embedding size.
        text_conditioning_tensor = self.text_linear_layer(text_encoder_output)

        # EXTRACT DOMINANT BIN FROM CHROMAGRAM

        # Original code
        # y, sr = sf.read(audio_file_path)
        # self.ver2(torch.from_numpy(y), sr)

        # Maffo's code
        y, sr = librosa.load(audio_file_path)
        chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
        dominant_bin = self.extract_dominant_bin(torch.tensor(chromagram, dtype=torch.float32))

        # Print chromagram
        # import matplotlib.pyplot as plt
        # chromagram2 = np.reshape(chromagram, chromagram.shape[:-1])
        # fig, ax = plt.subplots(nrows=2, sharex=True)
        # img = librosa.display.specshow(chromagram2, y_axis='chroma', x_axis='time', ax=ax[1])
        # fig.colorbar(img, ax=[ax[1]])
        # plt.show()

        # Expand the dimensions of the dominant bin tensor to match the text conditioning tensor
        dominant_bin = dominant_bin.unsqueeze(0).expand(text_conditioning_tensor.shape[:-1] + (-1,))

        return dominant_bin

        # # Concatenate text and melody conditioning tensors
        # concatenated_conditioning_tensor = torch.cat((text_conditioning_tensor, dominant_bin), dim=2)
        #
        # return concatenated_conditioning_tensor[0]


c = TextMelodyToTokenConverter()
c1 = c.convert_text_melody_to_tokens(
    text='salvatore is in the car',
    audio_file_path='/Users/salvatore/Desktop/Università/Development/NN/MusicGen/dataset/music_data/-0Gj8-vB1q4.wav'
)
