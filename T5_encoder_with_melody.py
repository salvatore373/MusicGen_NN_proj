import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import librosa
import numpy as np

class TextMelodyToTokenConverter:
    def __init__(self):
        self.text_linear_layer = torch.nn.Linear(768, 4)  # Assuming the original dimension is 768 for text

        for param in self.text_linear_layer.parameters():
            param.requires_grad_(False)

    def extract_dominant_bin(self, chromagram):
        # Assuming chromagram is a 2D tensor with shape (number of bins, number of frames)
        dominant_bin = torch.argmax(chromagram, dim=0)  # Select the dominant bin for each frame
        return dominant_bin

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
        y, sr = librosa.load(audio_file_path)
        chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
        dominant_bin = self.extract_dominant_bin(torch.tensor(chromagram, dtype=torch.float32))

        # Expand the dimensions of the dominant bin tensor to match the text conditioning tensor
        dominant_bin = dominant_bin.unsqueeze(0).expand(text_conditioning_tensor.shape[:-1] + (-1,))

        # Concatenate text and melody conditioning tensors
        concatenated_conditioning_tensor = torch.cat((text_conditioning_tensor, dominant_bin), dim=2)

        return concatenated_conditioning_tensor[0]
