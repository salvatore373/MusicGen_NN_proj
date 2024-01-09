import torch
import torch.nn as nn
import torch.nn.functional as F

from Encodec import get_tokens_from_file
from T5_encoder import TextToTokenConverter
from transformer import Transformer, TransformerWithText


class TransformerTrainer:
    def __init__(self, model: Transformer):
        """

        :param model: The model to train
        """
        self.model = model

    """
    def train(self, encoder_input, decoder_input, num_epochs: int, learning_rate: float):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-9)
        # loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            torch.cuda.empty_cache()
            self.model.train()

            # encoder_input = batch['encoder_input'].to(device)  # (b, seq_len)
            # decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = self.model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
            decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input,
                                               decoder_mask)  # (B, seq_len, d_model)

            # Compare the output with the label
            label = batch['label'].to(device)  # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # TODO: implement: Run validation at the end of every epoch
            # run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,
            #                lambda msg: batch_iterator.write(msg), global_step, writer)

            # Save the model at the end of every epoch
            # model_filename = ''
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'global_step': global_step
            # }, model_filename)
    """

    def train2(self, encoder_input, decoder_input, num_epochs: int, learning_rate: float,
               text_melody_conditioning=None):
        """
        :param text_melody_conditioning: The sequence of tokens to use for text or melody conditioning. It is
        strictly related to the model passed to this class' constructor.
        :return:
        """

        # Define the loss function and the optimizer to use to train
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Set the model in training mode
        self.model.train()

        for epoch in range(num_epochs):
            # Reset the gradients
            optimizer.zero_grad()
            # Calculate the output of the model
            output = self.model(encoder_input, decoder_input, text_melody_conditioning)
            # Compute the current loss
            # could use output.size(-1) instead of trg_vocab_size
            # todo: consider cross_entropy defined in official at 228
            loss = loss_fn(output, encoder_input)
            # Compute the gradients
            loss.backward()
            # Perform an optimization step
            optimizer.step()
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")


if __name__ == "__main__":
    # n x d
    # train_data = torch.rand(7, 4)
    train_data = get_tokens_from_file(
        '/Users/salvatore/Desktop/Università/Development/NN/MusicGen/dataset/music_data/-0Gj8-vB1q4.wav')
    train_data = train_data.t()
    train_data = train_data.to(torch.float32)

    enc_input = train_data
    # Add a 0-row for padding in order to match the size of the decoder input with the size
    # of the encoder one
    dec_input = F.pad(input=enc_input[1:], pad=(0, 0, 1, 0), mode='constant', value=0)

    # model_to_train = 'only_audio'
    model_to_train = 'with_text'
    # model_to_train = 'with_melody'

    if model_to_train == 'only_audio':
        model = Transformer(
            num_layers=5,
            q_val=4,
            v_val=4,
            dropout=0.1,
            ff_units=500,
            embed_size=train_data.shape[1],  # 4,
            trg_vocab_size=4,
            src_pad_idx=0,
        )
        trainer = TransformerTrainer(model)

        trainer.train2(enc_input, dec_input, num_epochs=5000, learning_rate=0.0001)
    elif model_to_train == 'with_text':
        text_descr = 'This is a description'
        text_tokens = TextToTokenConverter().convert_text_to_tokens(text_descr)

        model = TransformerWithText(
            num_layers=5,
            q_val=4,
            v_val=4,
            dropout=0.1,
            ff_units=500,
            embed_size=train_data.shape[1],  # 4,
            trg_vocab_size=4,
            src_pad_idx=0,
        )
        trainer = TransformerTrainer(model)

        trainer.train2(enc_input, dec_input, num_epochs=5000, learning_rate=0.0001,
                       text_melody_conditioning=text_tokens)
