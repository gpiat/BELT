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

    def __init__(self, tokenizer, n_classes, embed_size, nhead,
                 nhid, nlayers, phrase_len, dropout=0.5, pad_token=0):
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
                Defaults to 0 (same as BertTokenizer)
        """
        super(BELT, self).__init__()
        self.model_type = 'Transformer'
        self.phrase_len = phrase_len
        self.pad_token = pad_token
        self.numheads = nhead
        self.n_classes = n_classes
        self.tokenizer = tokenizer

        self.encoder = nn.Embedding(len(tokenizer.vocab), embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead,
                                                 nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(embed_size, n_classes)

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
            We want all tokens to be able to attend to themselves
            We want non-padding tokens to be able to attend to all the
            non-padding tokens
            We want padding tokens to attend to only themselves
            # combining these properties:

            [a b c <pad> <pad>] != <pad> -> [1 1 1 0 0]
            [1 1 1 0 0] * [[1 1 1 0 0]]  ->          ||| this matrix |||
                                                     vvv             vvv
                a     b     c   <pad> <pad>       a     b     c   <pad> <pad>
               ---------------------------       ---------------------------
            x | 1     0     0     0     0     x | 1     1     1     0     0
              |                                 |
            y | 0     1     0     0     0     y | 1     1     1     0     0
              |                                 |
            z | 0     0     1     0     0  +  z | 1     1     1     0     0
              |                                 |
            0 | 0     0     0     1     0     0 | 0     0     0     0     0
              |                                 |
            0 | 0     0     0     0     1     0 | 0     0     0     0     0

                                           =

                a     b     c   <pad> <pad>
               ---------------------------
            x | 2     1     1     0     0
              |
            y | 1     2     1     0     0
              |
            z | 1     1     2     0     0
              |
            0 | 0     0     0     1     0
              |
            0 | 0     0     0     0     1
        """

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

    def forward(self, batch):
        src = batch.get('input_ids')

        # src shape: torch.Size([minibatch, max_seq_len])
        # we want [minibatch, phrase_len] so we add a vertical strip of padding
        missing_cols = self.phrase_len - src.shape[1]
        assert missing_cols >= 0
        padding = torch.zeros(src.shape[0], missing_cols) + self.pad_token
        padding = padding.to(device)
        src = torch.cat((src, padding), 1)

        # From documentation:
        # key_padding_mask – if provided, specified padding elements in
        # the key will be ignored by the attention. This is a binary mask.
        # When the value is True, the corresponding value on the attention
        # layer will be filled with -inf.

        # File "torch/nn/functional.py", line 3330,
        #               in multi_head_attention_forward:
        # assert key_padding_mask.size(0) == bsz
        # so mask has to be shaped like src
        mask = (src == self.pad_token).to(device)

        # From the Embedding documentation examples:
        # >>> # a batch of 2 samples of 4 indices each
        # >>> input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
        # so input shape should be (minibatch, window_size)
        # here .long() is necessary because apparently torch.embedding (which
        # is called later on) requires the input tensor to be of type long
        output = self.encoder(src.long()) * math.sqrt(self.embed_size)
        # TODO: wait where does this * sqrt(embed size) come from?
        output = self.pos_encoder(output)
        # output shape: torch.Size([minibatch, window_size, embed_size])
        # as stated in this post:
        # https://discuss.pytorch.org/t/nn-transformer-explaination/53175/7
        # we should have torch.Size([window_size, minibatch, embed_size])
        output = output.permute(1, 0, 2)
        output = self.transformer_encoder(output,
                                          src_key_padding_mask=mask)
        # self._generate_mask(src))
        # output shape: torch.Size([minibatch, window_size, embed_size])

        output = self.decoder(output)
        # output shape: torch.Size([window_size + 1, minibatch, C])
        # TODO: wait, why window_size + 1?
        # with C the number of classes for the classification problem

        # To perform Softmax properly, torch.nn.CrossEntropyLoss expects
        # a tensor of shape [minibatch, C, window_size], yet we have
        # [window_size, minibatch, C]. Therefore we must permute dimensions
        # 2, 3 and 1.
        output = output.permute(1, 2, 0)
        # output shape: [minibatch, C, window_size]

        return output

    def filter(self, output, batch):
        """ This function takes a padded output of shape
            (batch_size, n_classes, phrase_len), finds and extracts only the
            predictions corresponding to the beginning of a token, and re-pads
            what's left to the shape (batch_size, n_classes, max_seq_len).
        """
        filtered = []
        for a, i in zip(output, batch.get("token_starts")):
            # We select relevant outputs, i.e. the ones that correspond to the
            # beginning of a token.
            selected = torch.index_select(a, 0, i)

            # We pad everything once again
            if selected.size(0) < batch.get("max_seq_len"):

                padding = torch.zeros(
                    batch.get("max_seq_len") -
                    selected.size(0), selected.size(1)
                ) + self.pad_token

                if device.type == 'cuda':
                    padding = padding.cuda()

                selected = torch.cat([selected, padding], 0)

            # We unsqueeze and append
            filtered.append(selected.unsqueeze(0))

        # We concatenate everything
        filtered = torch.cat(filtered, 0)
        return filtered

    def recycle(self, new_n_classes, initrange=0.1):
        self.n_classes = new_n_classes
        self.decoder = nn.Linear(self.embed_size, self.n_classes + 1)
        self._init_decoder_weights(initrange)
