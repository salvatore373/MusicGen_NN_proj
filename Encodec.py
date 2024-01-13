import torch
from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor
import soundfile as sf
import librosa


class AudioTokenizer:
    def __init__(self):
        self.model = EncodecModel.from_pretrained("facebook/encodec_24khz")
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

    def get_tokens_from_file(self, file_path):
        # Load the audio:
        audio_sample, sample_rate = sf.read(file_path)

        # Convert from multi-channel to mono-channel with the mean:
        if len(audio_sample.shape) > 1:
            # Se l'audio è stereo o multicanale, calcola la media dei canali per ottenere un segnale mono
            audio_sample = audio_sample.mean(axis=1)

        # Resampling:
        if sample_rate != self.processor.sampling_rate:
            audio_sample = librosa.resample(audio_sample, orig_sr=sample_rate, target_sr=self.processor.sampling_rate)

        inputs = self.processor(raw_audio=audio_sample, sampling_rate=self.processor.sampling_rate, return_tensors="pt")

        encoder_outputs = self.model.encode(input_values=inputs["input_values"],
                                            padding_mask=inputs.get("attention_mask", None),
                                            bandwidth=3.0)

        # Take the tokens with the attribute audio_codes
        tokens = encoder_outputs.audio_codes
        return tokens[0][0]

    @staticmethod
    def perform_quantization(tokens):
        outmap_min, _ = torch.min(tokens, dim=0, keepdim=True)
        outmap_max, _ = torch.max(tokens, dim=0, keepdim=True)
        normalized_audio_tokens = (tokens - outmap_min) / (outmap_max - outmap_min)  # Broadcasting rules apply

        # Define the number of quantization levels (bins)
        num_bins = 1024
        return (normalized_audio_tokens * (num_bins - 1)).to(torch.int32)

    def save_tokens_to_audio_file(self, tokens, output_file_path):
        tokens = AudioTokenizer.perform_quantization(tokens)
        tokens = tokens.unsqueeze(dim=0).unsqueeze(dim=0)
        audio_values = self.model.decode(tokens, [None], None)[0]

        # Convert the tensor to obtain a correct file audio wav
        reconstructed_audio = audio_values.detach().numpy().flatten()
        reconstructed_audio = reconstructed_audio * (2 ** 15)  # Scale the audio

        # Save the audio
        sf.write(output_file_path, reconstructed_audio, self.processor.sampling_rate)
