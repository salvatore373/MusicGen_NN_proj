from math import sqrt

import torch
import torch.nn as nn


class MaskedSelfAttention(nn.Module):
    def __init__(self,
                 q_param,
                 v_param):
        """
        Builds a block that computes masked self-attention.
        The q and v parameters
        :param q_param: The not-fixed dimension of the Query, Key matrices
        :param v_param: The not-fixed dimension of the Value matrix
        """
        super(MaskedSelfAttention, self).__init__()

        self.q_param = q_param
        self.v_param = v_param

        # considering a X of dimension (n, d):
        self.query = nn.Linear(q_param, q_param)  # output dim (n ,q)
        self.key = nn.Linear(q_param, q_param)  # output dim (n ,q)
        self.value = nn.Linear(v_param, v_param)  # output dim (n ,v)

        # Output linear layer (as indicated in Fig. 2 of "Attention is all you need")
        self.output_lin = nn.Linear(v_param, v_param)

    def forward(self, query, key, value, mask):
        """
        Compute masked self-attention for the given tuple of Query, Key and Value
        :return: The computed masked self-attention.
        """

        # Compute the values for query, key and value using learned params
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # todo: if the weights of Q, K, V are negative, than the input of the softmax can contain inf,
        #  resulting in a nan value in the corresponding output cell.

        # Compute Q * K^T
        query_key = torch.matmul(query, key.t())

        # Add masking
        masked = torch.mul(query_key, mask)

        # Compute the attention (with softmax along rows)
        h = torch.softmax(masked / sqrt(self.q_param), dim=1)
        h = torch.matmul(h, value)

        # Compute the output (dim = n,v)
        return self.output_lin(h)


class TransformerBlock(nn.Module):
    def __init__(self, q_val: int, v_val: int, dropout, ff_units):
        """
        Builds a block of the encoder part of the transformer
        :param q_val: The q parameter of the Self-Attention block.
        :param v_val: The v parameter of the Self-Attention block.
        :param dropout: The percentage of dropout to place after the normalization
        :param ff_units: The number of hidden units of the Feed Forward layer.
        """
        super(TransformerBlock, self).__init__()

        self.attention = MaskedSelfAttention(q_val, v_val)
        self.normalization_1 = nn.LayerNorm(v_val)

        # Define the FeedForward network as stated in section 3.3 of "Attention is all you need"
        self.feed_forward = nn.Sequential(
            nn.Linear(v_val, ff_units),
            nn.ReLU(),
            nn.Linear(ff_units, v_val),
        )

        self.normalization_2 = nn.LayerNorm(v_val)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        # Compute the masked self-attention
        attention = self.attention(query, key, value, mask)
        # Make layer normalization (with residual connection)
        first_normalization = self.dropout(self.normalization_1(attention + query))

        # Compute the feed forward
        feed_forward_out = self.feed_forward(first_normalization)
        # Make the second layer normalization (with residual connection)
        return self.dropout(self.normalization_2(feed_forward_out + first_normalization))


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, q_val: int, v_val: int, dropout, ff_units):
        """
        Builds the encoder part of a Transformer (a concatenation of TransformerBlocks)
        :param num_layers: The number of TransformerBlocks to include in the encoder.
        :param q_val: The q parameter of the Self-Attention block.
        :param v_val: The v parameter of the Self-Attention block.
        :param dropout: The percentage of dropout to place after the normalizations.
        :param ff_units: The number of hidden units to use in the TransformerBlock.
        """
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(TransformerBlock(q_val, v_val, dropout, ff_units))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """

        :param x: Should have the following shape: (num_codebooks, f_r)
        :param mask:
        :return:
        """
        # Note:
        # The input tokens are retrieved from the compression model, that have already imprinted the positional
        # encoding. For this reason they can be passed directly to the transformer.

        last_output = x
        for layer in self.layers:
            last_output = layer(last_output, last_output, last_output, mask)

        return last_output


class DecoderBlock(nn.Module):
    def __init__(self,
                 q_val: int,
                 v_val: int,
                 dropout: float,
                 ff_units: int):
        """
        :param q_val: The q parameter of the Self-Attention block.
        :param v_val: The v parameter of the Self-Attention block.
        :param dropout: The percentage of dropout to place after the normalizations.
        :param ff_units: The number of hidden units to use in the TransformerBlock.
        """
        super(DecoderBlock, self).__init__()
        self.attention = MaskedSelfAttention(q_val, v_val)
        self.normalization = nn.LayerNorm(v_val)
        self.transformer_block = TransformerBlock(q_val, v_val, dropout, ff_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        """

        :param x:
        :param value: The value matrix of the Self-Attention block, retrieved from the Transformer Encoder,
        to pass to a Self-Attention block in the decoder.
        :param key: The key matrix of the Self-Attention block, retrieved from the Transformer Encoder,
        to pass to a Self-Attention block in the decoder.
        :param src_mask: The mask passed to the encoder.
        :param trg_mask: The mask to use in the decoder.
        :return:
        """
        attention = self.attention(x, x, x, trg_mask)
        normalization = self.normalization(attention + x)
        transformer_block_output = self.transformer_block(normalization, key, value, src_mask)
        return transformer_block_output


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, q_val: int, v_val: int, dropout: float, ff_units: int, embed_size: int,
                 trg_vocab_size: int):
        """
        Builds the decoder part of a Transformer (a concatenation of DecoderBlocks)
        :param num_layers: The number of DecoderBlocks to include in the decoder.
        :param q_val: The q parameter of the Self-Attention block.
        :param v_val: The v parameter of the Self-Attention block.
        :param dropout: The percentage of dropout to place after the normalizations.
        :param ff_units: The number of hidden units to use in the TransformerBlock.
        :param embed_size: The size of the tokens passed to the Encoder.
        :param trg_vocab_size: The number of outputs of the decoder. This can also be intended as the number of
         output units of the decoder.
        """
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([DecoderBlock(q_val, v_val, dropout, ff_units) for _ in range(num_layers)])

        self.full_conn_out = nn.Sequential(
            nn.Linear(embed_size, trg_vocab_size),
            nn.Softmax(dim=1),  # todo: check value of dim
        )

    def forward(self, x, encoder_output, src_mask, trg_mask):
        """

        :param x:
        :param encoder_output: The output of the Encoder Block
        :param src_mask: The src_mask argument of the DecoderBlock
        :param trg_mask: The trg_mask argument of the DecoderBlock
        """
        curr_output = x
        for dec_block in self.layers:
            curr_output = dec_block(x, encoder_output, encoder_output, src_mask, trg_mask)

        return self.full_conn_out(curr_output)


class Transformer(nn.Module):
    def __init__(self, num_layers, q_val: int, v_val: int, dropout: float, ff_units: int, embed_size: int,
                 trg_vocab_size: int, src_pad_idx: int):
        """
        Builds a complete Transformer with an Encoder and a Decoder part
        :param num_layers: The number of layers to include in the encoder and in the decoder.
        :param q_val: The q parameter of the Self-Attention block.
        :param v_val: The v parameter of the Self-Attention block.
        :param dropout: The percentage of dropout to place after the normalizations.
        :param ff_units: The number of hidden units to use in the TransformerBlock.
        :param embed_size: The size of the tokens passed to the Encoder.
        :param trg_vocab_size: The number of outputs of the decoder. This can also be intended as the number of
         output units of the decoder.


        """
        super(Transformer, self).__init__()

        self.src_pad_idx = src_pad_idx
        self.trg_vocab_size = trg_vocab_size

        self.encoder = TransformerEncoder(num_layers, q_val, v_val, dropout, ff_units)
        self.decoder = TransformerDecoder(num_layers, q_val, v_val, dropout, ff_units, embed_size, trg_vocab_size)

    # TODO:
    def make_src_mask(self, src):
        # (N, 1, 1, src_len)
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    # TODO:
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        return torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

    def make_mask(self, dim):
        mask_ind = torch.tril(torch.ones((dim, dim), dtype=torch.bool), diagonal=-1).t()
        mask = torch.tril(torch.ones(dim, dim))
        mask[mask_ind] = float('-inf')
        return mask

    def forward(self, enc_input, dec_input):
        # Build masks for the encoder and for the decoder
        enc_mask = self.make_mask(enc_input.shape[0])
        dec_mask = self.make_mask(dec_input.shape[0])

        encoder_output = self.encoder(enc_input, enc_mask)
        return self.decoder(dec_input, encoder_output, enc_mask, dec_mask)


if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test the transformer passing example data as input
    x = torch.tensor([[1.0, 5.0, 6.0, 4.0, ], [1.0, 8.0, 7.0, 3.0, ]])  # n x d

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(
        num_layers=3,
        q_val=4,
        v_val=4,
        dropout=0.1,
        ff_units=500,
        embed_size=4,
        trg_vocab_size=4,
        src_pad_idx=0,
    )

    # The encoder should receive the input sequence, while the decoder the output/to-learn
    # sequence starting from index 1. In this case de decoder wants to reconstruct the input,
    # then pass a sequence to the encoder and the same sequence to the decoder without
    # the first element.
    out = model(x, x[1:])
    model.train()
    print(out.shape)
