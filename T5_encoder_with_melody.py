import torch
import torchaudio
import librosa
import numpy as np
import soundfile as sf


class MelodyToTokenConverter:

    def convert(self, audio_time_series, sr, embedding_size, argmax=False):
        audio_time_series = audio_time_series.to(torch.float)
        nfft = 128
        spec = torchaudio.transforms.Spectrogram(n_fft=nfft, power=2, center=True,
                                                 pad=0, normalized=True)
        from librosa import filters
        fbanks = torch.from_numpy(filters.chroma(sr=sr, n_fft=nfft, n_chroma=embedding_size))

        spec = spec(torch.transpose(audio_time_series, 0, 1)).squeeze(1)
        raw_chroma = torch.einsum('cf,...ft->...ct', fbanks, spec)
        norm_chroma = torch.nn.functional.normalize(raw_chroma, dim=-2, eps=1e-6)
        from einops import rearrange
        norm_chroma = rearrange(norm_chroma, 'b d t -> b t d')

        if argmax:
            idx = norm_chroma.argmax(-1, keepdim=True)
            norm_chroma[:] = 0
            norm_chroma.scatter_(dim=-1, index=idx, value=1)

        return norm_chroma

    def convert_text_melody_to_tokens(self, audio_file_path, embedding_size):
        y, sr = sf.read(audio_file_path)
        if len(y.shape) > 1:  # convert to mono from stereo
            y = y.mean(axis=1).reshape(-1, 1)
        res = self.convert(torch.from_numpy(y), sr, embedding_size)
        return res[0]
