import string


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


def peek(file, n=1):
    """ Reads n lines ahead in a file without changing the file
        object's current position in the file
        Args:
            - file: a text file
            - n (int): how many lines aheahd should be read, defaults to 1
        Return:
            - line: line read
    """
    pos = file.tell()
    lines = [file.readline() for i in range(n)]
    file.seek(pos)
    return lines[-1]


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

    def __init__(self, title, abstract, umls_mentions, no_punct=False):
        """ Args:
                - (str) title: raw title line of text
                - (str) abstract: raw abstract line of text
                - (list<str>) umls_mentions: list of raw lines
                    of text containing the UMLS entity mentions.
                - no_punct (bool): Defaults to False. If True,
                    removes punctuation from the `text` attribute.
                    This will cause training to fail as UMLS mentions
                    will no longer align with the text.
        """
        self.pmid, self.title = text_preprocess(title)
        _, self.abstract = text_preprocess(abstract)
        # no space is insterted between title and abstract to match up
        # with MedMentions PubTator format.
        self.raw_text = self.title + self.abstract

        self.no_punct = no_punct
        if no_punct:
            title_nopunct = self.title.translate(
                str.maketrans('', '', string.punctuation))
            abs_nopunct = self.abstract.translate(
                str.maketrans('', '', string.punctuation))
            # can't split raw_text because of missing
            # space between title and abstract
            self.text = [*title_nopunct.split(), *abs_nopunct.split()]
        else:
            self.text = [*self.title.split(), *self.abstract.split()]
        self.umls_entities = [UMLS_Entity(entity) for entity in umls_mentions]

    def get_cuid(self, word_idx):
        """ Returns the CUID of a word given its index (returns
            [None? 0? TODO] if not part of a UMLS concept mention)
            Args:
                word_idx (int): index of the word in self.text
        """
        # The idea here is to find the CUID of a word despite only having the
        # CUID of spans of characters. We therefore find the CUID of the last
        # character of the word. Since we have only the index of the word (and
        # not its characters), we have to apply a conversion by summing the
        # lengths of the words that come before it.
        char_idx = sum([len(word) for word in self.text[:(word_idx + 1)]])
        # can't forget to add the spaces
        # one between each word = (word_idx + 1) - 1
        # +1 because lists start at 0; -1 because one space between each word
        # for some reason this selects the space after the word, so I'm
        # substracting 1 but I can't figure out why
        char_idx += word_idx - 1
        cuid = None
        for mention in self.umls_entities:
            if mention.start_idx <= char_idx < mention.stop_idx:
                cuid = mention.concept_ID
                break  # it may theoretically be possible that one word is part
                # of several UMLS mentions but that case would be impractical
                # to handle and likely wouldn't matter at scale.
        return cuid

    def get_vocab(self):
        return list(set(self.text))

    def write_to(self, filename):
        with open(filename, 'a') as f:
            f.write('|'.join([self.pmid, 't', self.title]) + '\n')
            f.write('|'.join([self.pmid, 'a', self.abstract]) + '\n')
            for entity in self.umls_entities:
                f.write(str(entity) + '\n')
            f.write('\n')


class MedMentionsCorpus:
    """ This class instantiates a MedMentions corpus using one or more
        PubTator-formatted files. Its main purpose is to iterate over
        documents with the documents() generator function.
    """

    def __init__(self, fnames, auto_looping=False, no_punct=False):
        """ Args:
                - fnames (list<str>): list of filenames in the corpus
                - auto_looping (bool): whether retrieving documents should
                    automatically loop or not
                - no_punct (bool): Defaults to False. If True, removes
                    punctuation from each document's `text` attribute.
                    This will cause training to fail as UMLS mentions
                    will no longer align with the text.
        """
        self._filenames = fnames
        self._currentfile = 0
        self._looping = auto_looping
        self.no_punct = no_punct
        self.n_documents, self.cuids, self.vocab = self._get_cuids_and_vocab()
        self.nconcepts = len(self.cuids)

    def _get_cuids_and_vocab(self):
        """ Collects the CUIDs and vocabulary of the corpus.
            Should not be used outside of the constructor, because
            it relies on the document counter being at the start.
            If you need CUIDs or vocab, use the appropriate attributes.
        """
        cuids = set()
        vocab = set()
        n_documents = 0
        for document in self.documents():
            n_documents += 1
            for entity in document.umls_entities:
                cuids.add(entity.concept_ID)
            for word in document.text:
                vocab.add(word)
        self.loop_documents()
        return n_documents, cuids, vocab

    def documents(self):
        """ Yields:
                - pmid (str): the next document's PMID
                - title (str): the next document's title
                - abstract (str): the next document's abstract
                - umls_entities (list<str>): list of UMLS entities
                    for the next document
        """
        while self._currentfile < len(self._filenames):
            # opening the file -- not using `with` because
            # it causes excessive indentation
            f = open(self._filenames[self._currentfile], 'r')

            next_line = None
            while next_line != '' and peek(f) != '':
                title = f.readline()
                abstract = f.readline()
                next_line = f.readline()

                # after the abstract, each entity mention is written on a
                # separate line. The next document comes after a newline.
                umls_entities = []
                while next_line != '\n' and next_line != '':
                    # [:-1] deletes the trailing newline character
                    umls_entities.append(next_line[:-1])
                    next_line = f.readline()

                yield MedMentionsDocument(title, abstract,
                                          umls_entities, self.no_punct)
            f.close()

            self._currentfile += 1
            if self._currentfile >= len(self._filenames) and self._looping:
                self.loop_documents()
                return

    def loop_documents(self):
        """ Restarts the document file counter. This only takes
            effect after the file currently being read ends.
        """
        self._currentfile = 0
