import torch
import librosa


class MelodyConditioningModel:
    @staticmethod
    def extract_dominant_bin(chromagram):
        # Assuming chromagram is a 2D tensor with shape (number of bins, number of frames)
        bin_d = torch.argmax(chromagram, dim=0)  # Select the dominant bin for each frame
        return bin_d

    @staticmethod
    def process_chromagram(chromagram):
        # Apply normalization
        chromagram = torch.nn.functional.normalize(chromagram, dim=0, eps=1e-6)

        # Apply information bottleneck (commented out in the original code)
        # if True:
        if False:
            idx = chromagram.argmax(dim=0, keepdim=True)
            chromagram[:] = 0
            chromagram.scatter_(dim=0, index=idx, value=1)
        return chromagram

    def process_audio_file(self, audio_file_path):
        y, sr = librosa.load(audio_file_path)
        chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
        chromagram_tensor = torch.tensor(chromagram, dtype=torch.float32)

        dominant_bin_d = self.extract_dominant_bin(chromagram_tensor)
        chromagram_p = self.process_chromagram(chromagram_tensor)

        return dominant_bin_d, chromagram_p


model = MelodyConditioningModel()
dominant_bin, processed_chromagram = model.process_audio_file(r"C:\Users\matti\Downloads\electro_1.wav")

print("Dominant Bin:", dominant_bin)
print("Processed Chromagram:", processed_chromagram)
