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
        div_term = (torch.exp(torch.arange(0, d_model, 2).float() *
                              (-math.log(10000.0) / d_model)
                              )
                    )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, ntoken, n_umls_concepts, embed_size,
                 nhead, nhid, nlayers,  # batch_size,
                 phrase_len, dropout=0.5):
        """Args:
            - <int> ntoken: vocabulary size
            - <int> n_umls_concepts: number of UMLS concepts to choose from
            - <int> embed_size: embedding size
            - <int> nhead: number of heads in the multiheadattention models
            - <int> nhid: dimension of the FFNN in the encoder
            - <int> nlayers: number of TransformerEncoderLayer in encoder
            - <int> batch_size: number of phrases processed simultaneously
            - <int> phrase_len: number of words in each phrase processed
            - <float> dropout
        """
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        # self.src_mask = None
        self.encoder = nn.Embedding(ntoken, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead,
                                                 nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(embed_size, n_umls_concepts + 1)
        self.decoder2 = nn.Linear(phrase_len + 1, 1)
        self.softmax = nn.LogSoftmax(dim=0)

        self.embed_size = embed_size
        self.init_weights()

    # def _generate_square_subsequent_mask(self, sz):
    #     mask = (torch.tril(torch.ones(sz, sz), diagonal=(-1)))
    #     mask = mask.masked_fill(mask == 1, float('-inf'))
    #     return mask

    def init_weights(self, initrange=0.1):
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder2.bias.data.zero_()
        self.decoder2.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, target_words):
        # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #     device = src.device
        #     mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #     self.src_mask = mask

        src_l = src.tolist()
        target_words_list = target_words.tolist()
        target_word_indices = [src_l[i].index(
            target_words_list[i]) for i in range(len(src_l))]

        # print("1: ", src.shape)
        src = self.encoder(src) * math.sqrt(self.embed_size)
        # print("2: ", src.shape)
        src = self.pos_encoder(src)
        # print("3: ", src.shape)

        src_l = src.tolist()
        target_word_embeddings = torch.Tensor([src_l[i][target_word_indices[i]]
                                               for i in range(len(src_l))]
                                              ).to(device)
        target_word_embeddings = target_word_embeddings.unsqueeze(1)

        output = self.transformer_encoder(src)  # , self.src_mask)
        # print("4: ", output.shape)
        output = self.decoder(
            torch.cat((output, target_word_embeddings), dim=1))
        # print("5: ", output.shape)
        output = self.decoder2(output.transpose(1, 2)).squeeze(2)
        # print("6: ", output.shape)
        return output
