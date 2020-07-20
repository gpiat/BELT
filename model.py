import math
import torch
import torch.nn as nn

from constants import device

from torch.nn.modules import TransformerEncoder
from torch.nn.modules import TransformerEncoderLayer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = (torch.exp(torch.arange(0, d_model, 2).float()
                              * (-math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class BELT(nn.Module):

    def __init__(self, ntoken, n_classes, embed_size, nhead,
                 nhid, nlayers, phrase_len, dropout=0.5, pad_token=1):
        """Args:
            - <int> ntoken: vocabulary size
            - <int> n_classes: number of classes (such as UMLS concepts)
                that each token can fall into, not including the default
                class.
            - <int> embed_size: embedding size
            - <int> nhead: number of heads in the multiheadattention models
            - <int> nhid: dimension of the FFNN in the encoder
            - <int> nlayers: number of TransformerEncoderLayer in encoder
            - <int> phrase_len: number of words in each phrase processed
            - <float> dropout
            - pad_token: the number associated with the padding token.
                Defaults to 1 (assuming 0 is reserved for <UNK>)
        """
        super(BELT, self).__init__()
        self.model_type = 'Transformer'
        self.phrase_len = phrase_len
        self.pad_token = pad_token
        self.numheads = nhead
        self.n_classes = n_classes

        self.encoder = nn.Embedding(ntoken, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead,
                                                 nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(embed_size, n_classes + 1)

        self.embed_size = embed_size
        self.init_weights()

    def _generate_mask(self, src):
        """ Generating the mask which prevents the attention heads from
            focusing on the padding tokens.
            Example:
                The following matrix describes which part of the input are
                forcefully ignored (T) or taken into account normally (F).
                What we want is to attend to the the text when handling
                text and attend to nothing when handling padding. (only
                pertinent information is that the character is padding)

                in: [a, b, c, <pad>, <pad>]
                out: [x, y, z, 0, 0]
                      a     b     c   <pad> <pad>
                     ---------------------------
                  x | F     F     F     T     T
                    |
                  y | F     F     F     T     T
                    |
                  z | F     F     F     T     T
                    |
                  0 | T     T     T     F     T
                    |
                  0 | T     T     T     T     F
        """
        # We want all tokens to be able to attend to themselves
        # We want non-padding tokens to be able to attend to all the
        # non-padding tokens
        # We want padding tokens to attend to only themselves
        # # combining these properties:
        #
        # [a b c <pad> <pad>] != <pad> -> [1 1 1 0 0]
        # [1 1 1 0 0] * [[1 1 1 0 0]]  ->          ||| this matrix |||
        #                                          vvv             vvv
        #     a     b     c   <pad> <pad>       a     b     c   <pad> <pad>
        #    ---------------------------       ---------------------------
        # x | 1     0     0     0     0     x | 1     1     1     0     0
        #   |                                 |
        # y | 0     1     0     0     0     y | 1     1     1     0     0
        #   |                                 |
        # z | 0     0     1     0     0  +  z | 1     1     1     0     0
        #   |                                 |
        # 0 | 0     0     0     1     0     0 | 0     0     0     0     0
        #   |                                 |
        # 0 | 0     0     0     0     1     0 | 0     0     0     0     0

        #                                =

        #     a     b     c   <pad> <pad>
        #    ---------------------------
        # x | 2     1     1     0     0
        #   |
        # y | 1     2     1     0     0
        #   |
        # z | 1     1     2     0     0
        #   |
        # 0 | 0     0     0     1     0
        #   |
        # 0 | 0     0     0     0     1

        # creating a matrix of ones of size batch_size x window_size x 1
        # used to duplicate the identity matrix for each batch
        _BxWx1 = torch.ones(src.unsqueeze(2).size())
        # creating the identity matrix but with 1 matrix per batch
        identity_tensor = torch.eye(src.shape[1]).unsqueeze(0) * _BxWx1
        identity_tensor = identity_tensor.to(device)

        # creating the mask but missing padding tokens allowed to attend
        # to themselves
        mask = (src != self.pad_token).float().to(device)
        # mask.shape = src.shape = [batch_size, window_size]
        mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        # mask.shape = src.shape = [batch_size, window_size, window_size]
        mask = (mask + identity_tensor)
        # now mask has batch_size layers which all look like the last
        # matrix in the comment block at the beginning of this function

        # here, we're switching 1s and 0s and turning them into booleans
        mask = (1 - mask).clamp(min=0).bool()

        # attn_mask:
        #    2D or 3D mask that prevents attention to certain positions.
        #    A 2D mask will be broadcasted for all the batches while a 3D mask
        #    allows to specify a different mask for the entries of each batch.
        # attn_mask:
        #    3D mask :math:`(N*num_heads, L, S)` where N is the batch size,
        #    L is the target sequence length, S is the source sequence length.
        #    attn_mask ensures that position i is allowed to attend the
        #    unmasked positions.If a BoolTensor is provided, positions with
        #    ``True`` are not allowed to attend while ``False`` values will be
        #    unchanged.

        # N = minibatch = src.shape[0]
        # num_heads = self_attn.num_heads
        # L = window_size = src.shape[1]
        # S = window_size = src.shape[1]
        return mask

    def init_weights(self, initrange=0.1):
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self._init_decoder_weights(initrange)

    def _init_decoder_weights(self, initrange):
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # src shape: torch.Size([minibatch, window_size])
        output = self.encoder(src) * math.sqrt(self.embed_size)
        # output shape: torch.Size([minibatch, window_size, embed_size])
        output = self.pos_encoder(output)

        mask = (src != self.pad_token).to(device).transpose(0, 1)
        output = self.transformer_encoder(output,
                                          src_key_padding_mask=mask)
        # self._generate_mask(src))
        # output shape: torch.Size([minibatch, window_size, embed_size])

        output = self.decoder(output)
        # output shape: torch.Size([minibatch, window_size + 1, C])
        # with C the number of classes for the classification problem

        # To perform Softmax properly, torch.nn.CrossEntropyLoss expects
        # a tensor of shape [minibatch, C, window_size], yet we have
        # [minibatch, window_size, C]. Therefore we must permute dimensions
        # 1 and 2.
        output = output.permute(0, 2, 1)
        # output shape: [minibatch, C, window_size]
        return output

    def recycle(self, new_n_classes, initrange=0.1):
        self.n_classes = new_n_classes
        self.decoder = nn.Linear(self.embed_size, self.n_classes + 1)
        self._init_decoder_weights(initrange)
