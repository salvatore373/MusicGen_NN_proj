import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Transformer


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

    def train2(self, encoder_input, decoder_input, num_epochs: int, learning_rate: float):
        # Define the loss function and the optimizer to use to train
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Set the model in training mode
        self.model.train()

        for epoch in range(num_epochs):
            # Reset the gradients
            optimizer.zero_grad()
            # Calculate the output of the model
            output = self.model(encoder_input, decoder_input)
            # Compute the current loss
            # could use output.size(-1) instead of trg_vocab_size
            # todo: consider cross_entropy defined in official at 228
            loss = loss_fn(output.contiguous(), encoder_input.contiguous())
            # Compute the gradients
            loss.backward()
            # Perform an optimization step
            optimizer.step()
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")


if __name__ == "__main__":
    model = Transformer(
        num_layers=1,
        q_val=4,
        v_val=4,
        dropout=0.1,
        ff_units=10,
        embed_size=4,
        trg_vocab_size=4,
        src_pad_idx=0,
    )
    trainer = TransformerTrainer(model)

    # n x d -> 7 x 4
    train_data = torch.rand(7, 4)
    enc_input = train_data
    # Add a 0-row for padding in order to match the size of the decoder input with the size
    # of the encoder one
    dec_input = F.pad(input=train_data[1:], pad=(0, 0, 1, 0), mode='constant', value=0)
    trainer.train2(enc_input, dec_input, num_epochs=5000, learning_rate=0.0001)
