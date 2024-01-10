import librosa
import torch
import torch.nn.functional as F
from librosa import filters
import librosa
from einops import rearrange


class MelodyClass:
    def ver2_librosa(self, audio_times, sr, embedding_size):
        audio_times = audio_times.to(torch.float)
        y = audio_times.numpy()  # Converti il tensore in un array NumPy
        n_fft = 128
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, power=2, center=True, pad_mode='constant',
                                              norm='slaney')
        chroma_filter = filters.chroma(sr=sr, n_fft=n_fft, n_chroma=embedding_size)
        filter_tensor = torch.from_numpy(chroma_filter)
        spec = torch.tensor(spec, dtype=torch.float32).transpose(0, 1).unsqueeze(1)
        raw_chroma = torch.einsum('cf,...ft->...ct', filter_tensor, spec)
        norm_chroma = F.normalize(raw_chroma, dim=-2, eps=1e-6)
        norm_chroma = rearrange(norm_chroma, 'b d t -> b t d')

        return norm_chroma

    def convert_text_melody_to_tokens_librosa(self, audio_file_path, embedding_size):
        y, sr = librosa.load(audio_file_path, mono=True)
        res = self.ver2_librosa(torch.from_numpy(y), sr, embedding_size)
        return res[0]


if __name__ == "__main__":
    chroma_try = MelodyClass()

    audio_file_path = r"C:\Users\matti\Downloads\electro_1.wav"
    embedding_size = 4

    result_tensor = chroma_try.convert_text_melody_to_tokens_librosa(audio_file_path, embedding_size)

    print("Tensor shape:", result_tensor.shape)
    print(result_tensor)
