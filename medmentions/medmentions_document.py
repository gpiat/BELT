import itertools

from diff_match_patch import diff_match_patch
from util import TokenType


def text_preprocess(text):
    """ Takes the raw title and abstract as written in the file
        and keeps only the relevant text and PMID.
        Args:
            text (str): raw text of the form "PMID|?|Text\n"
                with '?' being 't' or 'a' depending on whether
                the text is a title or abstract.
        return:
            pmid (str): PMID of the article, of 1 to 8 digits
            text (str): cleaned text
    """
    # separating PMID and letter from the text.
    # *text captures the rest of the list in case
    # there happens to be a | in the text.
    pmid, _, *text = text.split("|")
    # joining the various fields of the title in case there
    # happens to be a | in the title text; removing \n
    text = "|".join(text)[:-1]

    return pmid, text


class UMLS_Entity:
    """ This data structure allows easy access to the various fields
        which describe a mention of a UMLS entity in a text.
    """

    def __init__(self, text):
        self._orig_text = text
        (_, self.start_idx, self.stop_idx, self.mention_text,
         self.semantic_type_ID, self.concept_ID) = text.split('\t')

        self.concept_ID = self.concept_ID[6:]
        self.start_idx = int(self.start_idx)
        self.stop_idx = int(self.stop_idx)

    def __str__(self):
        return self._orig_text


class MedMentionsDocument:
    """ This class instantiates a document from the MedMentions corpus
        using the information provided in the PubTator format.
        Attr:
            pmid (str): PMID of the document
            title (str): Title of the article
            abstract (str): Abstract of the article
            text (list<str>): title fused with abstract but as a list of words
            umls_entities (list<UMLS_Entity>): list of UMLS entity
                mentions in the text
            raw_text (str): simple concatenation of title and abstract. The
                indexing of characters in raw_text matches the one used in
                PubTator entity mention annotations.
    """

    def __init__(self, title, abstract, umls_mentions, tokenizer=None):
        """ Args:
                - (str) title: raw title line of text
                - (str) abstract: raw abstract line of text
                - (list<str>) umls_mentions: list of raw lines
                    of text containing the UMLS entity mentions.
                - tokenization (str): can be 'char', 'naive', or 'wordpiece'.
                    Determines how text is tokenized.
        """
        self.pmid, self.title = text_preprocess(title)
        _, self.abstract = text_preprocess(abstract)
        # no space is insterted between title and abstract to match up
        # with MedMentions PubTator format.
        self.raw_text = self.title + '\n' + self.abstract

        self.tokenizer = tokenizer
        self.text = self.tokenizer.tokenize(self.raw_text)
        self.tokenization = tokenizer.tokenization
        self.targets = self._initialize_targets()

        self.umls_entities = [UMLS_Entity(entity) for entity in umls_mentions]

        # list of all start and end indices of all entities
        # originally the stop index is exclusive, but we need it
        # to be inclusive and vice-versa for the start index.
        self.start_end_indices = list(itertools.chain(
            [(e.start_idx - 1, e.stop_idx - 1) for e in self.umls_entities]))

    def _initialize_targets(self, mode="cuid"):
        char_level_targets = [None] * len(self.raw_text)
        for i in range(len(char_level_targets)):
            for e in self.umls_entities:
                if i >= e.start_idx and i < e.stop_idx:
                    if mode == "cuid":
                        char_level_targets[i] = e.concept_ID
                    else:
                        char_level_targets[i] = e.semantic_type_ID
                    continue
                elif i > e.stop_idx:
                    continue

        self.targets = []
        token = iter(self.text)
        concat_tokens = ''.join(self.text)
        dmp = diff_match_patch()
        diff = dmp.diff_main(self.raw_text, concat_tokens)
        # this diff library creates diffs of the form:
        #   [(flag, substring), (flag, substring), ...]
        # where "flag" can be 1, -1 or 0 depending on whether
        # the substring is in concat_tokens but not raw_text,
        # vice-versa, or in both repectively.
        # example:
        #   raw_text = 'Nonylphenol'
        #   concat_tokens = 'Non##yl##phe##no##l'
        #   diff = [(0, 'Non'), (1, '##'), (0, 'yl'), (1, '##'),
        #           (0, 'phe'), (1, '##'), (0, 'no'), (1, '##'), (0, 'l')]
        # However, it is much easier to handle a character-level diff, e.g. :
        #   [(0, 'N'), (0, 'o'), (0, 'n'), (1, '#'), (1, '#'), (0, 'y'), ...]
        # This is what we set out to do with the following list
        # comprehension, which may seem obscure, but it is 30%
        # faster than the equivalent loop, which can be written as:
        # new_diff = []
        # for flag, sub_str in diff:
        #     new_diff += list(zip([flag] * len(sub_str), sub_str))
        # diff = new_diff
        #
        # diff = list(itertools.chain(*[zip([a] * len(b), b)
        #                               for a, b in diff]))
        #
        # actually we dont need to keep the character, further dividing
        # execution time by 2
        diff = list(itertools.chain(*[[flag] * len(sub_str)
                                      for flag, sub_str in diff]))

        token_targets = [None] * len(self.text)
        chars_left_in_current_token = len(next(token))
        current_char_index = 0
        current_token_index = 0
        for flag in diff:
            if chars_left_in_current_token == 0:
                token_targets[current_token_index] =\
                    char_level_targets[current_char_index]
                current_token_index += 1
                chars_left_in_current_token = len(next(token))
            if flag == 0:
                current_char_index += 1
                chars_left_in_current_token -= 1
            elif flag == -1:
                current_char_index += 1
            else:
                chars_left_in_current_token -= 1
        self.char_level_targets = char_level_targets
        self.targets = token_targets

    def get_vocab(self):
        try:
            return self.vocab
        except AttributeError:
            self.vocab = list(set(self.text))
        return self.vocab

    def write_to(self, filename):
        with open(filename, 'a') as f:
            f.write('|'.join([self.pmid, 't', self.title]) + '\n')
            f.write('|'.join([self.pmid, 'a', self.abstract]) + '\n')
            for entity in self.umls_entities:
                f.write(str(entity) + '\n')
            f.write('\n')
