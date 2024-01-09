import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")


class TextToTokenConverter:
    def __init__(self):
        # Since we want the token's dimension to be 4, I use a linear function that reduce the dimension.
        self.linear_layer = torch.nn.Linear(768, 4)
        self.linear_layer.requires_grad_(False)

    def convert_text_to_tokens(self, text):
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        model = T5ForConditionalGeneration.from_pretrained("t5-base")

        # ENCODER PART WITH T5-MODEL
        input_ids = tokenizer(text, return_tensors="pt").input_ids

        with torch.no_grad():
            encoder_output = model.encoder(input_ids).last_hidden_state

        # Generate the conditioning tensor, having as dimension T_C * D
        # where T_C = length of the sequence and D = embedding size.
        conditioning_tensor = self.linear_layer(encoder_output)

        return conditioning_tensor[0]
