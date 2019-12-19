import torch
from constants import device
# from math import ceil


class Numericalizer:
    def __init__(self, vocab):
        self.vocab = {word: (number + 2) for number, word in enumerate(vocab)}
        self.vocab['<unk>'] = 0
        self.vocab['<pad>'] = 1

    def numericalize_text(self, text):
        """ maps a list of words to a unique list of integers"""
        numericalized = []
        for word in text:
            try:
                numericalized.append(self.vocab[word])
            except KeyError:
                numericalized.append(self.vocab['<unk>'])
        return numericalized


def prod(numbers):
    res = 1
    for i in numbers:
        res *= i
    return res


# def integerify(s):
#     """ maps a string to a unique integer.
#         Thanks to user poke on StackOverflow
#     """
#     return int.from_bytes(s.encode(), 'big')


# def stringify(n):
#     """ Reverses integerify.
#         Thanks to user poke on StackOverflow
#     """
#     return n.to_bytes(math.ceil(n.bit_length() / 8), 'big').decode()


# def numericalize_text(text):
#     """ maps a list of words to a unique list of integers"""
#     return [integerify(word) for word in text]


def get_text_window(text, window_size, start_index, end_index):
    """
    """
    # creating a zero-filled vector of the right size in
    # case the entire document is smaller than the vector
    data = torch.zeros(window_size, dtype=torch.long).to(device)
    data[0:end_index - start_index] =\
        torch.Tensor(text[start_index:end_index]).to(device)
    return data


def get_start_end_indices(i, text_len, window_size):
    """
    """
    # assert window_size < text_len
    # start_index = i - window_size // 2
    # end_index = i + ceil(window_size / 2)
    # if start_index < 0:
    #     # subtracting negative number = adding positive
    #     end_index -= start_index
    #     start_index = 0
    # elif end_index > text_len:
    #     start_index -= end_index - text_len
    #     end_index = text_len

    # slicing the data to have batches of window_size
    # length, if possible centered around i, but starting
    # and ending within the list itself
    start_index = max(min(i - (window_size // 2),
                          text_len - window_size),
                      0)
    end_index = min(max(i + (window_size // 2),
                        window_size),
                    text_len)
    return start_index, end_index
