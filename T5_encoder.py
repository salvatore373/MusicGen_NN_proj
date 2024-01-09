import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

text1 = "translate English to German: The house is wonderful."
text2 = "translate English to German: The house is wonderful."

input_sequences = [text1, text2]

# Since we want the token's dimension to be 4, use a linear function that reduces the dimension.
linear_layer = torch.nn.Linear(768, 4)

# Tensors list
conditioning_tensors = []

# Through the input_sequences
for input_sequence in input_sequences:
    # ENCODER PART WITH T5-MODEL for each sentences
    input_ids = tokenizer(input_sequence, return_tensors="pt").input_ids
    with torch.no_grad():
        encoder_output = model.encoder(input_ids).last_hidden_state

    # Generate the conditioning tensor, having as dimension T_C * D
    # where T_C = length of the sequence and D = embedding size.
    conditioning_tensor = linear_layer(encoder_output)

    # add the conditioning tensor to the list
    conditioning_tensors.append(conditioning_tensor)

# Print the shapes and the tensors
for i, tensor in enumerate(conditioning_tensors):
    print(f"Dimensione di C per la frase {i + 1}: {tensor.shape}")
    print(f"Valori di C per la frase {i + 1}: {tensor}")

