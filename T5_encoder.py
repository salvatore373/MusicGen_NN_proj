import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


def c_tensor():
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    text = "translate English to German: The house is wonderful."

    # ENCODER PART WITH T5-MODEL
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    with torch.no_grad():
        encoder_output = model.encoder(input_ids).last_hidden_state

    # Since we want the token's dimension to be 4, I use a linear function that reduce the dimension.
    linear_layer = torch.nn.Linear(768, 4)

    # Generate the conditioning tensor, having as dimension T_C * D
    # where T_C = length of the sequence and D = embedding size.
    conditioning_tensor = linear_layer(encoder_output)

    print("C dimension: ", conditioning_tensor.shape)
    # output: [1, 11, 4] -> 11 = T_C = the length of the input's text
    # ==> The sequence consists of 11 tokens or fragments of text.
    # -> 4 = D = embedding size.
    # ==> Each token in the sequence is represented by a 4-dimensional vector.

    # Corretto salvo?
    return conditioning_tensor
