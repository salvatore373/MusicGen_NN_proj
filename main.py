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
        self.query = nn.Linear(q_param, q_param)  # dim (n ,q)
        self.key = nn.Linear(q_param, q_param)  # dim (n ,q)
        self.value = nn.Linear(v_param, v_param)  # dim (n ,v)

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

        # Compute Q * K^T
        query_key = torch.matmul(query, key.t())

        # Add masking
        masked = torch.mul(query_key, mask)

        # Compute the attention (with softmax along rows) # todo: check softmax is along rows
        h = torch.softmax(masked / torch.sqrt(self.q_param), dim=0)
        h = torch.matmul(h, value)

        # Compute the output
        return self.output_lin(h)


class TransformerBlock(nn.Module):
    def __init__(self, q_val, v_val, dropout, ff_units):
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
    def __init__(self, num_layers, q_val, v_val, dropout, ff_units):
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
        pass


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    # trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)
    #
    # src_pad_idx = 0
    # trg_pad_idx = 0
    # src_vocab_size = 10
    # trg_vocab_size = 10
    # model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
    #     device
    # )
    # out = model(x, trg[:, :-1])
    # print(out.shape)
