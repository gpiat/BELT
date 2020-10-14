import itertools

# from difflib import ndiff as diff
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

        self.umls_entities = [UMLS_Entity(entity) for entity in umls_mentions]

        # list of all start and end indices of all entities
        # originally the stop index is exclusive, but we need it
        # to be inclusive and vice-versa for the start index.
        self.start_end_indices = list(itertools.chain(
            [(e.start_idx - 1, e.stop_idx - 1) for e in self.umls_entities]))

    def get_mention_id(self, token_idx, mode="cuid"):
        """ Returns the CUID of a word given its index (returns
            None if not part of a UMLS concept mention)
            Args:
                token_idx (int): index of the word in self.text
        """

        cuid = None
        semtypeid = None

        if token_idx >= len(self.text):
            return None

        if self.tokenization == TokenType.CHAR:
            char_idx = token_idx
            # s_e_i_copy = list(itertools.chain(
            #                  *self.start_end_indices.copy()))
            # s_e_i_copy.append(token_idx)
            # s_e_i_copy.sort()
            # # if the character is in a mention, its index will be between
            # # a start and a stop index. Since they alternate, all start
            # # indices are at even positions within the list, and all stop
            # # indices will be at odd positions. Therefore, the index of a
            # # character that is in a mention will be put at an odd position
            # # in the list. Conversely, a character that is not in a mention
            # # will be at an even position. Since index() returns the first
            # # occurence, if the character examined is equal to the start
            # # index, it will be found at an even position, therefore the
            # # start index of the mention has to be exclusive. The reverse
            # # is true for the stop index of the mention.
            # token_idx_idx = s_e_i_copy.index(token_idx)
            # is_in_mention = bool(token_idx_idx % 2)
            # if is_in_mention:
            #     mention = self.umls_entities[int(token_idx_idx / 2)]
            #     cuid = mention.concept_ID
            #     semtypeid = mention.semantic_type_ID
        elif self.tokenization == TokenType.NAIVE:
            # The idea here is to find the CUID of a word despite only
            # having the CUID of spans of characters. We therefore find
            # the CUID of a character of the word. Since we have only the
            # index of the word (and not its characters), we have to apply
            # a conversion by summing the lengths of the words that come
            # before it.
            char_idx = sum([len(word) for word in self.text[:(token_idx + 1)]])
            # can't forget to add the spaces
            # one between each word = (token_idx + 1) - 1
            # +1 because lists start at 0; -1 because one space between
            # each word for some reason this selects the space after the word,
            # so I'm substracting 1 but I can't figure out why
            char_idx += token_idx - 1
            # char_idx is the index of the last character. Because of the way
            # strings are  split, this can point to a punctuation mark.
            # we subtract half the word length to get a character approximately
            # in the middle of the word.
            char_idx -= len(self.text[token_idx]) / 2

        else:
            # TODO: actually the proper way of doing this would be to do an
            # equivalent of diff and use that to track the corresponding index
            # in the raw text as you progress through the tokens and assign
            # a class... I think? This idea has a linear complexity, whereas
            # the following has an n**2 complexity (approx. linear for each
            # token)

            # This version still needs to be called many times, but at least
            # doesn't use diff
            # # getting previous tokens including the one we want
            previous_tokens = self.text[:token_idx + 1]
            # removing double #s and reversing the list so that we can pop()
            # in the correct order
            previous_tokens = [wp if not wp.startswith('##') else wp[2:]
                               for wp in previous_tokens]
            previous_tokens.reverse()
            # at this point all we should need are to add the correct spaces
            # back in. The tokenizer removes all spaces, so we don't have that
            # to worry about.

            text_before = []
            original_text_counter = 0
            # we're going through the tokens, keeping an index to the original
            # text aligned with the characters in the tokens, and adding spaces
            # back in between the tokens where they're missing.
            for i in range(len(previous_tokens)):
                while self.raw_text[original_text_counter] == ' ':
                    text_before.append(' ')
                    original_text_counter += 1
                wp = previous_tokens.pop()
                original_text_counter += len(wp)
                text_before.append(wp)

            text_before = ''.join(text_before)

            # The following uses diff and thus was retired
            # # the difficulty here is that wordpiece adds characters that
            # # need to be accounted for when resolving character indices

            # # Given that the tokenized text contains artifacts such as `##`
            # # markers to denote suffixes, the idea is to let the tokenizer
            # # figure out what the text before the current token was like.
            # # We can then count characters to find the beginning and end
            # # indices of the characters of the current token.

            # # However, it's not as simple as that because the decoded text
            # # is not exactly the same as the original text.
            # # For starters, `[CLS] ` and ` [SEP]` are added at the beginning
            # # and end of the decoded text respectively. Removing these is
            # # simple enough.
            # #    (In fact, there may or may not be a space after `[CLS]`
            # #    depending on whether the first token is a suffix or not, but
            # #    this shouldn't come into consideration since the text won't
            # #    start with a suffix)

            # # Getting the text before the token we want.
            # # The [6:-6] is to remove the `[CLS] ` and ` [SEP]` markers.
            # # Here we choose to include the token of interest because it
            # # automatically handles the problem of knowing whether to
            # # account for a space before the token of interest.

            # text_before = self.tokenizer.decode(
            #     self.tokenizer.encode(self.text[:token_idx + 1]))[6:-6]

            # # Furthermore, when parsing punctuation, some extra whitespaces
            # # may be introduced. To remove these, we use difflib to identify
            # # these inaccuracies w.r.t. the original text.
            # text_before = ''.join([c[-1]
            #                        for c in diff(text_before,
            #                                      self.raw_text.lower())
            #                        if c[0] != '+'])

            # # char_idx is the index of the last character of the token.
            # # We assume there are no mentions that stop in the middle of a
            # # token, meaning that taking any character of the mention is
            # # fine.
            char_idx = len(text_before) - 1

        for mention in self.umls_entities:
            if mention.start_idx <= char_idx < mention.stop_idx:
                cuid = mention.concept_ID
                semtypeid = mention.semantic_type_ID
                break  # it may theoretically be possible that one word is
                # part of several UMLS mentions but that case would be
                # impractical to handle and likely wouldn't matter at
                # scale.

        if mode == "cuid":
            return cuid
        elif mode == "semtype":
            return semtypeid
        else:
            raise ValueError('Invalid argument for method get_mention_id '
                             ' of class MedMentionsDocument: "mode" argument '
                             'must either have value "cuid" or "semtype", but '
                             'found ' + str(mode))

    def get_mention_ids(self, first_token_idx, last_token_idx, mode="cuid"):
        """ Gets the ID of the entity for the corresponding
            mention of each token.
            Args:
                first_token_idx: index of the first token
                    to examine in self.text
                last_token_idx: index of the last token
                    to examine in self.text
                mode: one of "cuid" or "semtype" depending on whether the
                    semantic type ID or UMLS Concept Unique Identifier
                    should be returned.
        """
        return [self.get_mention_id(token_idx, mode=mode)
                for token_idx in range(first_token_idx, last_token_idx)]

    def _initialize_targets(self, mode="cuid"):
        self.targets = []

    def get_vocab(self):
        return list(set(self.text))

    def write_to(self, filename):
        with open(filename, 'a') as f:
            f.write('|'.join([self.pmid, 't', self.title]) + '\n')
            f.write('|'.join([self.pmid, 'a', self.abstract]) + '\n')
            for entity in self.umls_entities:
                f.write(str(entity) + '\n')
            f.write('\n')
