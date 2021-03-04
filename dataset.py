import copy
import numpy as np
import re
import torch

from torch.utils.data import Dataset
from transformers import BertTokenizer


# ## Label extraction ##
def extract_label_mapping(file_list: dict = None):
    """ We retrieve all categories in the corpus and create the tagging scheme
        mapping.
        Args:
            file_list (dict=None): actually not a list, but a dict mapping
                names to file names like: {"train":"trainfile.tsv", ...}
    """
    categories = set()

    # TODO: add a case if labels.txt is found?
    # For each set (train/dev/test)
    for corpus_part, filename in file_list.items():
        with open(filename, "r", encoding="UTF-8") as input_file:
            for line in input_file:
                # ignore empty lines
                if re.match("^$", line):
                    continue

                columns = line.rstrip("\n").split("\t")
                # The last column contains the category label
                # but might contain a "tag" prefixing the label
                categories.add(columns[-1])

    label_mapping = {label: i for i, label in enumerate(categories)}
    # adding the null category, for tokens that
    # do not belong to a given category.
    if "O" not in label_mapping:
        label_mapping["O"] = len(label_mapping)

    return label_mapping


# ## Dataset object creation ##
class NERDataset(Dataset):
    """ This object aims at making a corpus easier to handle.
        Attr:
            medmentions_file (str): filename for the corpus
            label_mapping (dict): maps class labels to numbers such as
                {'O'; 0, 'Disease': 1, 'Treatment': 2, ...}
            bert_tokenizer (BertTokenizer): not used for tokenization per se,
                but is required to shape inputs properly for processing by a
                BERT model
            instances (dict): maps indices to instances. Each "instance" is a
                dict containing information on a text sequence. The `instances`
                dict may look something like:
                {0: {"seq_input_ids": [101, 3455, 566, 554, 355, 667, 102],
                     "seq_token_starts": [1, 2, 4, 5],
                     "seq_labels": [0, 3, 0, 0]
                    },
                 1: ...}
                 See create_instance for explanations on what an instance is.
    """

    def __init__(self, medmentions_file: str = None,
                 bert_tokenizer: BertTokenizer = None,
                 label_mapping: dict = None,
                 max_seq_len=512):
        """ Args:
                medmentions_file (str=None): filename for the corpus
                bert_tokenizer (BertTokenizer=None): not used for tokenization
                    per se, but is required to shape inputs properly for
                    processing by a BERT model.
                label_mapping (dict=None): maps class labels to numbers such as
                    {'O'; 0, 'Disease': 1, 'Treatment': 2, ...}
        """

        self.medmentions_file = medmentions_file
        self.label_mapping = label_mapping
        self.bert_tokenizer = bert_tokenizer
        self.absolute_max_seq_len = max_seq_len

        self.instances = dict()

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances.get(idx)

    def load_instances(self):
        """ Load instances from file.
            The function processes the medmentions file line by line. Once it
            reaches a blank line or the end of the file, it creates an instance
            and adds it to the instance dictionary (self.instances)
        """

        instance_idx = 0
        instance_list = []

        with open(self.medmentions_file, "r", encoding="UTF-8") as input_file:
            sequence_buffer = []

            for line in input_file:
                # if line is empty
                if re.match("^$", line):
                    # create new instance with what we have so far and continue
                    if len(sequence_buffer) > 0:
                        instance = self.create_instance(
                            sequence_buffer=sequence_buffer)
                        instance_list.append(instance)
                        instance_idx += 1
                        sequence_buffer = []
                    continue

                sequence_buffer.append(line.rstrip("\n").split("\t"))

            if len(sequence_buffer) > 0:
                instance = self.create_instance(
                    sequence_buffer=sequence_buffer)
                instance_list.append(instance)
                instance_idx += 1
        i = 0
        while i < len(instance_list):
            instance = instance_list[i]
            if len(instance["seq_input_ids"]) > self.absolute_max_seq_len:
                beginning = instance_list[:i]  # current item i is excluded
                end = instance_list[i + 1:]
                instance_split = self.split_instance(instance)
                instance_list = beginning + instance_split + end
                i += 1
            i += 1
        self.instances = {k: v for k, v in enumerate(instance_list)}

    def create_instance(self, sequence_buffer: list = None):
        """ Encode and keep track of the sentence represented by the sequence
            buffer.
            Args:
                sequence_buffer (list=None): A list of indexable datastructures
                    of the form [subtoken, ..., label] where the ellipsis can
                    contain anything (it is ignored).
                    sequence_buffer example:
                    [["the", "O"], ["ele", "S-ANIMAL",],
                     ["##phant", "S-ANIMAL",], ["is", "O"], ["gray", "O"]]
            Return:
                An "instance", which is a dict that looks something like this:
                {"seq_input_ids": [101, 3455, 566, 554, 355, 667, 102],
                 "seq_token_starts": [1, 2, 4, 5],
                 "seq_labels": [0, 3, 0, 0]}
                In this example, the instance contains the encoded sentence
                "the elephant is gray", tokenized as ["[CLS]", "the", "ele",
                "##phant", "is", "gray", "[SEP]"] and encoded as the
                seq_input_ids field. The seq_token_starts field designates
                "the", "ele", "is", and "gray" as subtokens that begin a token
                in the sequence. The seq_labels field designates the "elephant"
                token as being a different type of token from the others, where
                0 may map to "O" (meaning "not an entity") and 3 may map to
                "S-ANIMAL" in label_mapping.
        """
        new_instance = dict()
        seq_token_lens = []
        token_labels = []

        # The first column of the medmentions file contains token forms
        # example: subtoken_forms = ["the", "ele", "##phant", "is", "gray"]
        subtoken_forms = [item[0] for item in sequence_buffer]

        # The last column of the medmentions file contains labels
        # ex: subtoken_labels = ["O", "S-ANIMAL", "S-ANIMAL", "O", "O"]
        subtoken_labels = [item[-1] for item in sequence_buffer]

        # We keep track of the token lengths
        # (i.e. how many pieces compose each token)
        # ex: seq_token_lens = [1, 2, 1, 1]
        for i in range(len(subtoken_forms)):
            subtoken = subtoken_forms[i]
            # note: re.match checks from the beginning of the string
            if re.match("##", subtoken) and len(seq_token_lens) > 0:
                seq_token_lens[-1] += 1
            else:
                seq_token_lens.append(1)
                # Isolating the label of the current subtoken because in this
                # case it is the label of an entire token (or the first label
                # of a series of subtokens that form the same token)
                # ex: token_labels = ["O", "S-ANIMAL", "O", "O"]
                token_labels.append(subtoken_labels[i])

        # We add the [CLS] and [SEP] tokens to the sentence
        # ex: seq_token_pieces = ["[CLS]", "the", "ele", "#phant",
        #                         "is", "gray", "[SEP]"]
        seq_token_pieces = [self.bert_tokenizer.cls_token] + \
            subtoken_forms + [self.bert_tokenizer.sep_token]

        # We keep the index of the starting piece for each token
        # in the sequence buffer.
        # ex: seq_token_starts = [1, 2, 4, 5]
        seq_token_starts = 1 + np.cumsum([0] + seq_token_lens[:-1])

        # Final step: we convert token pieces to their respective ids
        # seq_input_ids = [101, 3455, 566, 554, 355, 667, 102]
        seq_input_ids = self.bert_tokenizer.convert_tokens_to_ids(
            seq_token_pieces)

        # We store everything and return the new instance
        new_instance["seq_input_ids"] = seq_input_ids
        new_instance["seq_token_starts"] = seq_token_starts
        new_instance["raw_seq_labels"] = token_labels
        new_instance["seq_labels"] = [self.label_mapping.get(item)
                                      for item in token_labels]

        return new_instance

    def split_instance(self, instance):
        instance1 = dict()
        instance2 = dict()
        for i, ts in enumerate(instance["seq_token_starts"]):
            if ts > self.absolute_max_seq_len:
                break
        # ts is the first out of bounds token and i is its index in
        # seq_token_starts. However, the previous token may be
        # partially out of bounds as well. So we subtract 1 and that
        # index will be excluded when it is the end point of a slice.
        i -= 1
        ts = instance["seq_token_starts"][i]
        instance1["seq_input_ids"] = instance["seq_input_ids"][:ts]
        instance1["seq_token_starts"] = instance["seq_token_starts"][:i]
        instance1["raw_seq_labels"] = instance["raw_seq_labels"][:i]
        instance1["seq_labels"] = instance["seq_labels"][:i]

        instance2["seq_input_ids"] = instance["seq_input_ids"][ts:]
        # for instance 2 it's not enough to just take the symmetric slice,
        # we need to restart the indices from 0.
        instance2["seq_token_starts"] = \
            [sts - ts for sts in instance["seq_token_starts"][i:]]
        instance2["raw_seq_labels"] = instance["raw_seq_labels"][i:]
        instance2["seq_labels"] = instance["seq_labels"][i:]
        return [instance1, instance2]


# ## Dataloader creation ##
def collate_ner(batch, pad_id: int = 0):
    """ Create a mini-batch from a list of instances.
        Args:
            batch (iterable): contains instances (see NERDataset.load_instances
                for details) to create a minibatch that can be fed to a pytorch
                DataLoader.
            pad_id (int=0): ID of the token used for padding. Defaults to
                0 because that's actually the pad_id used by bert and the
                huggingface trainer doesn't know to specify that as an
                argument.
        Return:
            final_batch (dict): all relevant instance information formatted
                appropriately as tensors for use by a pytorch DataLoader
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    final_batch = dict()

    # Extracting useful information from batch instances
    # ==================================================
    all_seq_token_starts = []
    all_seq_input_ids = []
    all_seq_labels = []
    all_seq_raw_labels = []

    for instance in batch:
        all_seq_token_starts.append(instance.get("seq_token_starts"))
        all_seq_input_ids.append(instance.get("seq_input_ids"))
        all_seq_labels.append(instance.get("seq_labels"))
        all_seq_raw_labels.append(instance.get("raw_seq_labels"))

    final_batch["raw_seq_labels"] = all_seq_raw_labels

    # Extracting the len of the longest sequence of word pieces
    max_seq_len_pieces = max([len(seq) for seq in all_seq_input_ids])

    # Extracting the len of the longest sequence of tokens
    # Why "max_seq_len_SEQ"? => max sequence length among sequences of TOKENS
    #                           Which should be less than the max sequence
    #                           length of sequences of subtokens
    max_seq_len_seq = max([len(seq) for seq in all_seq_labels])

    final_batch["max_seq_len"] = max_seq_len_seq

    # Preparing BERT mini-batch
    # Three things are needed:
    # 1. input_ids
    # 2. input_type_ids
    # 3. input_masks
    # ==========================
    tensor_all_input_ids = []
    tensor_all_input_type_ids = []
    tensor_all_input_masks = []

    for cur_input_ids in all_seq_input_ids:
        # type_ids for the first sentence in the sequence must be 0
        # TOASK: why?
        cur_input_type_ids = [0] * len(cur_input_ids)
        # mask is set to 1 for useful pieces of the sequence,
        # in this case the instance sequence
        cur_input_mask = [1] * len(cur_input_ids)

        # Padding everything to max_seq_len_pieces
        while len(cur_input_ids) < max_seq_len_pieces:
            # PAD word piece ID provided as argument of the function
            cur_input_ids.append(pad_id)
            cur_input_type_ids.append(0)  # type_ids are padded with 0 for BERT
            cur_input_mask.append(0)  # We mask with 0

        # Converting everything to a LongTensor
        # and appending to their respective list
        try:
            tensor_all_input_ids.append(
                torch.LongTensor([cur_input_ids])
            )
            tensor_all_input_type_ids.append(
                torch.LongTensor([cur_input_type_ids])
            )
            tensor_all_input_masks.append(
                torch.LongTensor([cur_input_mask])
            )
        except TypeError as e:
            print("pad_id: ", pad_id)
            print("cur_input_ids: ", cur_input_ids)
            print("cur_input_type_ids: ", cur_input_type_ids)
            print("cur_input_mask: ", cur_input_mask)
            raise e

    # Concatenating everything and placing payloads into the final batch
    tensor_all_input_ids = torch.cat(tensor_all_input_ids, 0)
    tensor_all_input_type_ids = torch.cat(tensor_all_input_type_ids, 0)
    tensor_all_input_masks = torch.cat(tensor_all_input_masks, 0)

    final_batch["input_ids"] = tensor_all_input_ids.to(DEVICE)
    final_batch["input_type_ids"] = tensor_all_input_type_ids.to(DEVICE)
    final_batch["input_mask"] = tensor_all_input_masks.to(DEVICE)

    # Combining token starts
    # No need to concatenate everything
    # ===========================================================
    tensor_all_token_starts = []

    for seq_token_starts in all_seq_token_starts:
        tensor_all_token_starts.append(
            torch.LongTensor(seq_token_starts).to(DEVICE)
        )

    final_batch["token_starts"] = tensor_all_token_starts

    # LABELS
    # ============================================================
    tensor_all_labels = []

    for seq_labels in all_seq_labels:
        cur_labels = copy.deepcopy(seq_labels)

        # Padding with -100, the default "ignore" value in pytorch loss
        # functions making it unnecessary to mask loss.
        while len(cur_labels) < max_seq_len_seq:
            cur_labels.append(-100)

        tensor_all_labels.append(cur_labels)

    # Concatenating everything
    final_batch["token_labels"] =\
        torch.LongTensor(tensor_all_labels).to(DEVICE)

    # MASKS
    # =============================================================
    tensor_all_token_masks = []

    for instance in batch:
        cur_mask = []

        # Mask 1 for sequence tokens
        for _ in instance["seq_labels"]:
            cur_mask.append(1)

        # Mask 0 for padding elements
        while len(cur_mask) < max_seq_len_seq:
            cur_mask.append(0)

        tensor_all_token_masks.append(cur_mask)

    # Concatenating everything
    final_batch["token_masks"] =\
        torch.LongTensor(tensor_all_token_masks).to(DEVICE)

    return final_batch
