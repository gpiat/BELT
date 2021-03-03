import pickle
import torch

from constants import device
from model import BELT


def binary_target_finder(document,
                         start_index,
                         end_index,
                         umls_concepts):
    """ Finds the targets for a set of tokens when in "bin" mode.
        Args:
            document: the document being processed
            start_index: index of the first token
            end_index: index of the last token
            umls_concepts (dict): lookup table to translate
                CUIDs to their corresponding indices in the
                prediction vector for the token.
    """
    return [int(i is not None) for i in
            document.targets[start_index:end_index]]


def CUID_target_finder(document,
                       start_index,
                       end_index,
                       umls_concepts):
    """ Finds the targets for a set of tokens when in "cuid" mode.
        Args:
            document: the document being processed
            start_index: index of the first token
            end_index: index of the last token
            umls_concepts (dict): lookup table to translate
                CUIDs to their corresponding indices in the
                prediction vector for the token.
    """
    return [umls_concepts[cuid] for cuid, _ in
            document.targets[start_index:end_index]]


def get_text_window(text, window_size, start_index, end_index, pad_token=1):
    """
    """
    # creating a zero-filled vector of the right size in
    # case the entire document is smaller than the vector
    data = torch.full([window_size], pad_token, dtype=torch.long).to(device)
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


def load_model(args, target_indexing, tokenizer=None):
    """ Loads or creates a model for training.
        Args:
            args (dict): a dict of arguments processed from the command line
            target_indexing (dict): a lookup table for determining class names
            tokenizer: Optional because only useful if a new model is being
                created.
    """
    n_classes = len(target_indexing) if args['--target_type'] != "bin" else 1
    if args['--resume']:
        with open(args['--writepath'] + args['--model_fname'],
                  'rb') as model_file:
            model = pickle.load(model_file)
        # when transfer learning to a model with a different number
        # of classes, we replace the last layer of the model.
        if (model.decoder.out_features - 1) != n_classes:
            model.decoder = torch.nn.Linear(model.embed_size, n_classes + 1)
    else:
        model = BELT(tokenizer=tokenizer, n_classes=n_classes,
                     embed_size=args['--embed_size'], nhead=args['--nhead'],
                     nhid=args['--nhid'], nlayers=args['--nlayers'],
                     phrase_len=args['--window_size'],
                     dropout=args['--dropout']).to(device)
    return model


def semtype_target_finder(document,
                          start_index,
                          end_index,
                          sem_types):
    """ Finds the targets for a set of tokens when in "semantic type" mode.
        Args:
            document: the document being processed
            start_index: index of the first token
            end_index: index of the last token
            sem_types (dict): lookup table to translate
                STIDs to their corresponding indices in the
                prediction vector for the token.
    """
    return [sem_types[stid] for _, stid in
            document.targets[start_index:end_index]]


def pad(text, window_size, overlap, batch_size=1, pad_token='<pad>'):
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

    out_text += [pad_token] * pad_amount
    # Now we're actually not done. We need a to be a multiple of
    # batch_size.
    a = len(out_text) // (window_size - overlap)
    incomplete_batch_size = a % batch_size
    if incomplete_batch_size > 0:
        missing_windows = batch_size - incomplete_batch_size
        out_text += [pad_token] * (window_size - overlap) * missing_windows
    return out_text


def set_targets(target_type):
    if target_type == "cuid":
        return CUID_target_finder
    elif target_type == "semtype":
        return semtype_target_finder
    elif target_type == "bin":
        return binary_target_finder
