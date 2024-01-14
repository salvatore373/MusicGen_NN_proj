import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_preparation import load_dataset
from transformer import Transformer, TransformerWithText


class TransformerTrainer:
    def __init__(self, model: Transformer, model_save_dir):
        """

        :param model: The model to train
        :param model_save_dir: The directory where to save the trained model at the end of training.
        """
        self.model = model
        self.model_save_dir = model_save_dir

    def train_on_sample(self, encoder_input, decoder_input, num_epochs: int, learning_rate: float,
                        text_conditioning=None):
        """
        :param text_conditioning: The sequence of tokens to use for text conditioning. It is
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
            output = self.model(encoder_input, decoder_input, text_conditioning)
            # Compute the current loss
            loss = loss_fn(output, encoder_input)
            # Compute the gradients
            loss.backward()
            # Perform an optimization step
            optimizer.step()
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

    def train_on_sample_and_melody(self, encoder_input, decoder_input, num_epochs: int, learning_rate: float,
                                   text_conditioning=None, melody_conditioning=None):
        """
        :param text_conditioning: The sequence of tokens to use for text conditioning. It is
        strictly related to the model passed to this class' constructor.
        :param melody_conditioning: The sequence of tokens to use for melody conditioning. It is
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
            output = self.model(encoder_input, decoder_input, text_conditioning, melody_conditioning)
            # Compute the current loss
            loss = loss_fn(output, torch.cat((melody_conditioning, encoder_input), dim=0))
            # Compute the gradients
            loss.backward()
            # Perform an optimization step
            optimizer.step()
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

    def train_on_dataset(self, dataset, num_epochs: int, learning_rate: float):
        """
        Trains the model on the given dataset.
        :param dataset: As returned from data_preparation.load_dataset().
        """
        # Define the loss function and the optimizer to use to train
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Set the model in training mode
        self.model.train()

        N = len(dataset)
        for epoch in range(num_epochs):
            # Reset the gradients
            optimizer.zero_grad()

            losses = torch.zeros(N)
            for i in range(N):
                # Get a sequence of tokens representing an audio file, and the associated text conditioning
                audio = dataset[i]['audio']  # n1 x d
                text = dataset[i]['text']  # n2 x d

                enc_input = audio
                dec_input = F.pad(input=enc_input[1:], pad=(0, 0, 1, 0), mode='constant', value=0)
                # Calculate the output of the model
                output = self.model(enc_input, dec_input, text)

                # Compute and store the loss for the current sequence
                curr_loss = loss_fn(output, enc_input).reshape(1)
                losses[i] = curr_loss

            # Compute the sum of the losses for all the samples
            loss = torch.sum(losses)
            # Compute the gradients
            loss.backward()
            # Perform an optimization step
            optimizer.step()
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

        # Save the model
        model_name = {round(time.time() * 1000)}
        torch.save(self.model.state_dict(), f'{self.model_save_dir}/{model_name}')
        torch.save(self.model, f'{self.model_save_dir}/{model_name}-2')


if __name__ == "__main__":
    dataset = load_dataset()
    model = TransformerWithText(
        num_layers=5,
        q_val=4,
        v_val=4,
        h_val=3,
        dropout=0.1,
        embed_size=4,  # 4,
        trg_vocab_size=4,
        src_pad_idx=0,
    )
    trainer = TransformerTrainer(model, '/Volumes/SALVATORE R/Università/NN/models')
    trainer.train_on_dataset(dataset, num_epochs=100, learning_rate=1e-1)
