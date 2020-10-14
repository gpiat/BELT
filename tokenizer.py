import pickle

from enum import Enum
from transformers import BertTokenizer


class TokenType(Enum):
    NAIVE = 0
    WORD = 0
    CHARACTER = 1
    CHAR = 1
    WORDPIECE = 2
    WP = 2

    @classmethod
    def from_str(cls, s):
        s = s.lower()
        try:
            return cls(int(s))
        except ValueError:
            pass
        if s in ['naive', 'word']:
            return cls.NAIVE
        elif s in ['char', 'character']:
            return cls.CHAR
        elif s in ['wordpiece', 'wp']:
            return cls.WP
        else:
            raise ValueError("{} not recognized as a token type."
                             " Available token types are 'naive',"
                             " 'char' and 'wordpiece'.".format(s))


class BaseTokenizer:
    """ This is not a class that should be instantiated. Its encode() method
        calls the tokenize() method which is not declared, but should be
        implemented in subclasses.
    """

    def __init__(self, vocab_fname=None):
        """ Args:
                vocab_fname (str): optional. If given, the vocabulary will be
                loaded from the file.
        """
        self.tokenization = TokenType.CHAR
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.pad_token_id = 0
        self.unk_token_id = 1
        if vocab_fname is not None:
            with open(vocab_fname, 'rb') as f:
                vocab = pickle.load(f)
            self.init_vocab(vocab)

    def init_vocab(self, vocab):
        self.vocab = {word: (number + 2)
                      for number, word in enumerate(vocab)}
        self.vocab[self.unk_token] = self.unk_token_id
        self.vocab[self.pad_token] = self.pad_token_id
        self.vocab_size = len(self.vocab)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        if type(text) == str:
            text = self.tokenize(text)
        encoded = self._encode_decode(text, self.vocab, self.unk_token_id)
        return encoded

    def decode(self, encoded_text):
        decoded = self._encode_decode(
            encoded_text, self.ids_to_tokens, self.unk_token)
        decoded = self.detokenize(decoded)
        return decoded

    def _encode_decode(self, input_txt, converter, default_value):
        output_text = []
        for token in input_txt:
            try:
                output_text.append(converter[token])
            except KeyError:
                output_text.append(default_value)
        return output_text


class CharTokenizer(BaseTokenizer):
    def __init__(self, vocab_fname):
        super().__init__(vocab_fname)

    def tokenize(self, text):
        return list(text)

    def detokenize(self, tokenized_txt):
        return ''.join(tokenized_txt)


class NaiveTokenizer(BaseTokenizer):
    def __init__(self, vocab_fname):
        super().__init__(vocab_fname)

    def tokenize(self, text):
        return text.split()

    def detokenize(self, tokenized_txt):
        return ' '.join(tokenized_txt)


def get_tokenizer(tokenization, vocab_fname):
    if tokenization == TokenType.CHAR:
        return CharTokenizer(vocab_fname)
    elif tokenization == TokenType.WP:
        tok = BertTokenizer.from_pretrained(vocab_fname)
        tok.tokenization = TokenType.WP
        return tok
    else:
        return NaiveTokenizer(vocab_fname)


# class Numericalizer:
#     def __init__(self, corpus):
#         self.tokenization = corpus.tokenization
#         if corpus.tokenization == TokenType.WP:
#             self.tokenizer = corpus.tokenizer
#             self.vocab = self.tokenizer.vocab
#             self.pad_token = self.tokenizer.pad_token
#             self.pad_token_id = self.tokenizer.pad_token_id
#             self.unk_token = self.tokenizer.unk_token
#             self.unk_token_id = self.tokenizer.unk_token_id
#         else:
#             self.vocab = {word: (number + 2)
#                           for number, word in enumerate(corpus.vocab)}
#             self.unk_token = '<unk>'
#             self.pad_token = '<pad>'
#             self.unk_token_id = 0
#             self.pad_token_id = 1
#             self.vocab[self.unk_token] = self.unk_token_id
#             self.vocab[self.pad_token] = self.pad_token_id

#     def numericalize_text(self, text):
#         """ maps a list of tokens to a unique list of integers"""
#         if self.tokenization == TokenType.WP:
#             return self.tokenizer.encode(text)
#         # else:
#         numericalized = []
#         for token in text:
#             try:
#                 numericalized.append(self.vocab[token])
#             except KeyError:
#                 numericalized.append(self.vocab[self.unk_token])
#         return numericalized


# class FastTextVectorizer(Numericalizer):
#     # TODO
#     def __init__(self, vocab):
#         pass
