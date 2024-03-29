import sys
from math import sqrt

import torch
import torch.nn as nn

from T5_encoder import TextToTokenConverter


class MaskedSelfAttention(nn.Module):
    def __init__(self,
                 q_param,
                 v_param):
        """
        Builds a block that computes masked self-attention.
        :param q_param: The not-fixed dimension of the Query, Key matrices
        :param v_param: The not-fixed dimension of the Value matrix
        """
        super(MaskedSelfAttention, self).__init__()

        self.q_param = q_param
        self.v_param = v_param

        # The first dim of query, key and value Linear should be the embedding dimension.
        # Assuming for simplicity that it is equal to q and v

        # considering a X of dimension (n, d):
        self.query = nn.Linear(q_param, q_param)  # output dim (n ,q)
        self.key = nn.Linear(q_param, q_param)  # output dim (n ,q)
        self.value = nn.Linear(q_param, v_param)  # output dim (n ,v)

    def forward(self, query, key, value, mask=None):
        """
        Compute masked self-attention for the given tuple of Query, Key and Value
        :param query
        :param key
        :param value
        :param mask: The mask used in the masked-attention formula. If not given, a matrix of ones will be used.
        :return: The computed masked self-attention.
        """

        # Compute the values for query, key and value using learned params
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # Compute Q * K^T
        query_key = torch.matmul(query, key.t())

        # Add masking
        if mask is None:
            mask = torch.ones(query_key.size())
        masked = torch.mul(query_key, mask)

        # Compute the attention (with softmax along rows)
        h = torch.softmax(masked / sqrt(self.q_param), dim=1)
        return torch.matmul(h, value)


class CrossAttention(MaskedSelfAttention):
    """
    A block that computes cross-attention.
    """

    def forward(self, s1, s2):
        """
        Compute the cross-attention for the given input sequences
        :param s1: The sequence to use as query.
        :param s2: The sequence to use as key and value.
        :return: The cross-attention for the given input sequences.
        """
        return super().forward(s1, s2, s2)


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, h_param, q_param, v_param, *args, **kwargs):
        """
        Builds a block that computes masked self-attention a number of times equal to h_param.
        :param h_param: The number of times that masked self-attention should be performed.
        :param q_param: The not-fixed dimension of the Query, Key matrices
        :param v_param: The not-fixed dimension of the Value matrix
        """
        super(MaskedMultiHeadAttention, self).__init__()
        self.h_param = h_param
        self.q_param = q_param
        self.v_param = v_param

        self.attention_layers = [MaskedSelfAttention(q_param, v_param) for _ in range(h_param)]

        # Output linear layer (as indicated in Fig. 2 of "Attention is all you need")
        self.output_lin = nn.Linear(h_param * v_param, v_param)

    def forward(self, query, key, value, mask=None):
        # Compute the self-attention h times
        attentions = [self_att_layer(query, key, value, mask)
                      for self_att_layer in self.attention_layers]
        # Concatenate the result of each self-attention and then project them
        concatenation = torch.cat(attentions, dim=1)
        return self.output_lin(concatenation)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, h_param, q_param, v_param, *args, **kwargs):
        """
        Builds a block that computes masked self-attention a number of times equal to h_param.
        :param h_param: The number of times that masked self-attention should be performed.
        :param q_param: The not-fixed dimension of the Query, Key matrices
        :param v_param: The not-fixed dimension of the Value matrix
        """
        super(MultiHeadCrossAttention, self).__init__()
        self.h_param = h_param
        self.q_param = q_param
        self.v_param = v_param

        self.attention_layers = [CrossAttention(q_param, v_param) for _ in range(h_param)]

        # Output linear layer (as indicated in Fig. 2 of "Attention is all you need")
        self.output_lin = nn.Linear(h_param * v_param, v_param)

    def forward(self, s1, s2):
        # Compute the self-attention h times
        attentions = [cross_att_layer(s1, s2)
                      for cross_att_layer in self.attention_layers]
        # Concatenate the result of each self-attention and then project them
        concatenation = torch.cat(attentions, dim=1)
        return self.output_lin(concatenation)


class TransformerBlock(nn.Module):
    def __init__(self, q_val: int, v_val: int, h_val: int, dropout):
        """
        Builds a block of the encoder part of the transformer
        :param q_val: The q parameter of the Multi-Head-Attention block.
        :param v_val: The v parameter of the Multi-Head-Attention block.
        :param h_val: The v parameter of the Multi-Head-Attention block.
        :param dropout: The percentage of dropout to place after the normalization
        """
        super(TransformerBlock, self).__init__()

        self.attention = MaskedMultiHeadAttention(h_val, q_val, v_val)
        self.normalization_1 = nn.LayerNorm(v_val)

        # Define the FeedForward network as stated in section 3.3 of "Attention is all you need"
        self.feed_forward = nn.Sequential(
            nn.Linear(v_val, 4 * v_val),
            nn.ReLU(),
            nn.Linear(4 * v_val, v_val),
        )

        self.normalization_2 = nn.LayerNorm(v_val)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # Compute the masked self-attention
        attention = self.attention(query, key, value)
        # Make layer normalization (with residual connection)
        first_normalization = self.dropout(self.normalization_1(attention + query))

        # Compute the feed forward
        feed_forward_out = self.feed_forward(first_normalization)
        # Make the second layer normalization (with residual connection)
        return self.dropout(self.normalization_2(feed_forward_out + first_normalization))


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, q_val: int, v_val: int, h_val: int, dropout):
        """
        Builds the encoder part of a Transformer (a concatenation of TransformerBlocks)
        :param num_layers: The number of TransformerBlocks to include in the encoder.
        :param q_val: The q parameter of the Multi-Head-Attention block.
        :param v_val: The v parameter of the Multi-Head-Attention block.
        :param h_val: The h parameter of the Multi-Head-Attention block.
        :param dropout: The percentage of dropout to place after the normalizations.
        """
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(TransformerBlock(q_val, v_val, h_val, dropout))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """

        :param x: Should have the following shape: (num_codebooks, f_r)
        :return:
        """
        # Note:
        # The input tokens are retrieved from the compression model, that have already imprinted the positional
        # encoding. For this reason they can be passed directly to the transformer.

        last_output = x
        for layer in self.layers:
            last_output = layer(last_output, last_output, last_output)

        return last_output


class DecoderBlock(nn.Module):
    def __init__(self,
                 q_val: int,
                 v_val: int,
                 h_val: int,
                 dropout: float, ):
        """
        :param q_val: The q parameter of the Multi-Head-Attention blocks.
        :param v_val: The v parameter of the Multi-Head-Attention blocks.
        :param h_val: The h parameter of the Multi-Head-Attention blocks.
        :param dropout: The percentage of dropout to place after the normalizations.
        """
        super(DecoderBlock, self).__init__()
        self.attention = MaskedSelfAttention(q_val, v_val)
        self.normalization = nn.LayerNorm(v_val)
        self.transformer_block = TransformerBlock(q_val, v_val, h_val, dropout)
        self.dropout = nn.Dropout(dropout)
        self.cross_attention = MultiHeadCrossAttention(h_val, q_val, v_val)

    def forward(self, x, value, key, src_mask, trg_mask, text_tokens=None):
        """
        :param x: The input of the block.
        :param value: The value matrix of the Self-Attention block, retrieved from the Transformer Encoder,
        to pass to a Self-Attention block in the decoder.
        :param key: The key matrix of the Self-Attention block, retrieved from the Transformer Encoder,
        to pass to a Self-Attention block in the decoder.
        :param src_mask: The mask passed to the encoder.
        :param trg_mask: The mask to use in the decoder.
        :param text_tokens: The embedded text to use for text-conditioning. Cross-validation will be computed
         between text_tokens and the output of the block. If not provided, text-conditioning will not be performed.
        :return:
        """
        attention = self.attention(x, x, x, trg_mask)
        normalization = self.normalization(attention + x)
        transformer_block_output = self.transformer_block(normalization, key, value)
        if text_tokens is None:
            return transformer_block_output
        else:
            return self.cross_attention(transformer_block_output, text_tokens)


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, q_val: int, v_val: int, h_val: int, dropout: float, embed_size: int,
                 trg_vocab_size: int):
        """
        Builds the decoder part of a Transformer (a concatenation of DecoderBlocks)
        :param num_layers: The number of DecoderBlocks to include in the decoder.
        :param q_val: The q parameter of the Multi-Head-Attention block.
        :param v_val: The v parameter of the Multi-Head-Attention block.
        :param h_val: The h parameter of the Multi-Head-Attention block.
        :param dropout: The percentage of dropout to place after the normalizations.
        :param embed_size: The size of the tokens passed to the Encoder.
        :param trg_vocab_size: The number of outputs of the decoder. This can also be intended as the number of
         output units of the decoder.
        """
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([DecoderBlock(q_val, v_val, h_val, dropout) for _ in range(num_layers)])

        self.full_conn_out = nn.Sequential(
            nn.Linear(embed_size, trg_vocab_size),
            nn.Softmax(dim=1),
        )

    def forward(self, x, encoder_output, src_mask, trg_mask, text_tokens=None):
        """

        :param x:
        :param encoder_output: The output of the Encoder Block
        :param src_mask: The src_mask argument of the DecoderBlock
        :param trg_mask: The trg_mask argument of the DecoderBlock
        :param text_tokens: The embedded text to use for text-conditioning. If not provided, text-conditioning
         will not be performed.
        """
        curr_output = x
        for dec_block in self.layers:
            curr_output = dec_block(x, encoder_output, encoder_output, src_mask, trg_mask, text_tokens)

        return self.full_conn_out(curr_output)


class Transformer(nn.Module):
    def __init__(self, num_layers, q_val: int, v_val: int, h_val: int, dropout: float, embed_size: int,
                 trg_vocab_size: int, src_pad_idx: int):
        """
        Builds a complete Transformer with an Encoder and a Decoder part
        :param num_layers: The number of layers to include in the encoder and in the decoder.
        :param q_val: The q parameter of the Multi-Head-Attention block.
        :param v_val: The v parameter of the Multi-Head-Attention block.
        :param h_val: The h parameter of the Multi-Head-Attention block.
        :param dropout: The percentage of dropout to place after the normalizations.
        :param embed_size: The size of the tokens passed to the Encoder.
        :param trg_vocab_size: The number of outputs of the decoder. This can also be intended as the number of
         output units of the decoder.


        """
        super(Transformer, self).__init__()

        self.src_pad_idx = src_pad_idx
        self.trg_vocab_size = trg_vocab_size

        self.encoder = TransformerEncoder(num_layers, q_val, v_val, h_val, dropout)
        self.decoder = TransformerDecoder(num_layers, q_val, v_val, h_val, dropout, embed_size, trg_vocab_size)

    @staticmethod
    def make_mask(dim):
        mask_ind = torch.tril(torch.ones((dim, dim), dtype=torch.bool), diagonal=-1).t()
        mask = torch.tril(torch.ones(dim, dim))
        mask[mask_ind] = sys.float_info.min

        return mask

    def forward(self, enc_input, dec_input):
        # Build masks for the encoder and for the decoder
        enc_mask = self.make_mask(enc_input.shape[0])
        dec_mask = self.make_mask(dec_input.shape[0])

        encoder_output = self.encoder(enc_input)
        return self.decoder(dec_input, encoder_output, enc_mask, dec_mask)


class TransformerWithText(Transformer):
    """
    A Transformer that accepts text conditioning
    """

    def forward(self, enc_input, dec_input, text_tokens):
        # Build masks for the encoder and for the decoder
        enc_mask = self.make_mask(enc_input.shape[0])
        dec_mask = self.make_mask(dec_input.shape[0])

        encoder_output = self.encoder(enc_input)
        return self.decoder(dec_input, encoder_output, enc_mask, dec_mask, text_tokens)


class TransformerWithTextAndMelody(TransformerWithText):
    """
    Builds a Transformer that accepts both text and melody conditioning
    """

    def forward(self, enc_input, dec_input, text_tokens, melody_chromagram):
        # Compute the conditioned input of the decoder
        dec_input_conditioned = torch.concat((melody_chromagram, dec_input), dim=0)

        # Build masks for the encoder and for the decoder
        enc_mask = self.make_mask(enc_input.shape[0])
        dec_mask = self.make_mask(dec_input_conditioned.shape[0])

        encoder_output = self.encoder(enc_input)
        return self.decoder(dec_input_conditioned, encoder_output, enc_mask, dec_mask, text_tokens)


if __name__ == "__main__":
    # Test the transformer passing example data as input
    embedding_dim = 4
    x = torch.rand(7, embedding_dim)  # n x d

    # text_tokens = torch.rand(2, embedding_dim)
    text_descr = 'This is a description'
    text_tokens = TextToTokenConverter().convert_text_to_tokens(text_descr)  # n2 x d

    melody_tokens = torch.rand(15, embedding_dim)  # n3 x d

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    # model = TransformerWithTextAndMelody(
    model = TransformerWithText(
        # model = Transformer(
        num_layers=3,
        q_val=4,
        v_val=4,
        h_val=3,
        dropout=0.1,
        embed_size=embedding_dim,
        trg_vocab_size=4,
        src_pad_idx=0,
    )

    # The encoder should receive the input sequence, while the decoder the output/to-learn
    # sequence starting from index 1. In this case de decoder wants to reconstruct the input,
    # then pass a sequence to the encoder and the same sequence to the decoder without
    # the first element.
    # out = model(x, x[1:], text_tokens, melody_tokens)  # text and melody version
    out = model(x, x[1:], text_tokens)  # text version
    # out = model(x, x[1:])  # simple version
    model.train()
    print(out.shape)
