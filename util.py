import torch
import torch_optimizer
from constants import device
# from math import ceil


class Numericalizer:
    def __init__(self, vocab):
        self.vocab = {word: (number + 2) for number, word in enumerate(vocab)}
        self.vocab['<unk>'] = 0
        self.vocab['<pad>'] = 1

    def numericalize_text(self, text):
        """ maps a list of tokens to a unique list of integers"""
        numericalized = []
        for token in text:
            try:
                numericalized.append(self.vocab[token])
            except KeyError:
                numericalized.append(self.vocab['<unk>'])
        return numericalized


class FastTextVectorizer(Numericalizer):
    def __init__(self, vocab):
        pass


# def prod(numbers):
#     res = 1
#     for i in numbers:
#         res *= i
#     return res


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


def select_optimizer(option, model, lr):
    # optimizer selection

    if option == "adam":
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.8)
    elif option == "radam":
        optimizer = torch_optimizer.RAdam(model.parameters(),
                                          lr=0.001,
                                          betas=(0.9, 0.999),
                                          eps=1e-8,
                                          weight_decay=0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.8)
    elif option == "adamw":
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.8)
    elif option == "adamax":
        optimizer = torch.optim.Adamax(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.8)
    elif option == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.8)
    elif option == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.8)
    elif option == "adabound":
        optimizer = torch_optimizer.AdaBound(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            final_lr=0.1,
            gamma=1e-3,
            eps=1e-8,
            weight_decay=0,
            amsbound=False,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.8)
    elif option == "sparseadam":
        optimizer = torch.optim.SparseAdam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.8)
    elif option == "lbfgs":
        optimizer = torch.optim.LBFGS(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.2)
    elif option == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.2)
    elif option == "rprop":
        optimizer = torch.optim.Rprop(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.2)
    elif option == "lamb":
        optimizer = torch_optimizer.Lamb(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.8)
    elif option == "asgd":
        optimizer = torch.optim.ASGD(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.2)
    elif option == "accsgd":
        optimizer = torch_optimizer.AccSGD(
            model.parameters(),
            lr=1e-3,
            kappa=1000.0,
            xi=10.0,
            small_const=0.7,
            weight_decay=0
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.2)
    elif option == "sgdw":
        optimizer = torch_optimizer.SGDW(
            model.parameters(),
            lr=1e-3,
            momentum=0,
            dampening=0,
            weight_decay=1e-2,
            nesterov=False,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.2)
    elif option == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    10.0,
                                                    gamma=0.01)
    else:
        raise TypeError("optimizer name not recognized")
    return optimizer, scheduler


def pad(text, window_size, overlap, batch_size=1):
    """ Pads text with iterations of the '<pad>' token so that a whole number
        of batches of a whole number of overlapping windows fits in the length
        of the text.
        Args:
            text (list<str>): the text to pad
            window_size (int): the size of the windows of text that will be
                processed
            overlap (float[0,1]): the proportion of text that should overlap
                between windows. 0 = no overlap, 1 = the window will slide
                by half its size
            batch_size (int): number of windows to process simultaneously.
                defaults to 1.
    """
    out_text = text.copy()
    # The overlap argument is a proportion, we're turning it into an
    # actual number. Here, overlap refers to the proportion of a given
    # window that is shared with either the previous or the next window.
    # By this definition, 100% overlap is when the window slides by half
    # the window size.
    overlap = round(overlap * window_size / 2)
    # now, "overlap" refers to the amount of overlap between one window
    # and the next window. We want the text length to be of the form
    # `a*(window_size - overlap) + overlap` with `a` a constant
    # example: window_size = 5, overlap = 1, a = 3
    # wwww owww owww w     < w for elements in a window, o for overlaps
    # ---- =___ =--- -     < - for odd windows, _ for even ones,
    #                        = for overlaps
    # But first we need to handle the special case where the text is
    # shorter than a = 1
    pad_amount = 0
    if len(text) <= window_size:
        pad_amount = window_size - len(text)
    else:
        excess = (len(text) - overlap) % (window_size - overlap)
        if excess > 0:
            # the padding amount is however much you need to fill up a
            # window after accounting for excess and overlap
            pad_amount = window_size - overlap - excess

    out_text += ['<pad>'] * pad_amount
    # Now we're actually not done. We need a to be a multiple of
    # batch_size.
    a = len(out_text) // (window_size - overlap)
    incomplete_batch_size = a % batch_size
    if incomplete_batch_size > 0:
        missing_windows = batch_size - incomplete_batch_size
        out_text += ['<pad>'] * (window_size - overlap) * missing_windows
    return out_text
