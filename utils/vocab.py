import collections

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
SOS_WORD = '<s>'
EOS_WORD = '</s>'


class Vocabulary(object):
    def __init__(self, lower=False,
                 pad_word=None,
                 unk_word=None,
                 sos_word=None,
                 eos_word=None):

        self.idx2word = {}
        self.word2idx = {}
        self.frequencies = {}
        self.lower = lower

        # Special entries will not be pruned.
        self._PAD_WORD = pad_word if pad_word is not None else PAD_WORD
        self._UNK_WORD = unk_word if unk_word is not None else UNK_WORD
        self._SOS_WORD = sos_word if sos_word is not None else SOS_WORD
        self._EOS_WORD = eos_word if eos_word is not None else EOS_WORD

        # special words
        self.special_words = [self._PAD_WORD,
                              self._UNK_WORD,
                              self._SOS_WORD,
                              self._EOS_WORD]
        for special_word in self.special_words:
            self.add(special_word)

        self.special = []

    @property
    def sos_idx(self):
        return self.lookup(self._SOS_WORD)

    @property
    def eos_idx(self):
        return self.lookup(self._EOS_WORD)

    @property
    def pad_idx(self):
        return self.lookup(self._PAD_WORD)

    @property
    def size(self):
        return len(self.idx2word)

    def load_file(self, filename):
        """Load entries from a file."""
        for line in open(filename, 'r', encoding='utf-8'):
            fields = line.split()
            word = fields[0]
            idx = int(fields[1])
            self.add(word, idx)

    def write_file(self, filename):
        """Write entries to a file."""
        with open(filename, 'w', encoding='utf-8') as file:
            for i in range(self.size):
                word = self.idx2word[i]
                idx = self.word2idx[word]
                file.write('%s\t%d\n' % (word, idx))

    def lookup(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.word2idx[key]
        except KeyError:
            return default

    def get_word(self, idx, default=None):
        try:
            return self.idx2word[idx]
        except KeyError:
            return default

    def add_special(self, word, idx=None):
        """Mark this `word` and `idx` as special (i.e. will not be pruned)."""
        idx = self.add(word, idx)
        self.special += [idx]

    def add_specials(self, words):
        """Mark all words in `words` as specials (i.e. will not be pruned)."""
        for word in words:
            self.add_special(word)

    def add(self, word, idx=None):
        """Add `word` in the dictionary. Use `idx` as its index if given."""
        word = word.lower() if self.lower else word
        if idx is not None:
            self.idx2word[idx] = word
            self.word2idx[word] = idx
        else:
            if word in self.word2idx:
                idx = self.word2idx[word]
            else:
                idx = len(self.idx2word)
                self.idx2word[idx] = word
                self.word2idx[word] = idx

        if idx not in self.frequencies:
            self.frequencies[idx] = 1
        else:
            self.frequencies[idx] += 1

        return idx

    def prune(self, size):
        """Return a new dictionary with the `size` most frequent entries."""
        if size >= self.size:
            return self

        # Only keep the `size` most frequent entries.
        freqs = [[i, self.frequencies[i]] for i in range(self.size)]
        sorted_freqs = sorted(freqs, key=lambda it: it[1], reverse=True)

        new_vocab = Vocabulary()
        new_vocab.lower = self.lower

        # Add special entries in all cases.
        for i in self.special:
            new_vocab.add_special(self.idx2word[i])

        for i, _ in sorted_freqs[:size]:
            new_vocab.add(self.idx2word[i])

        return new_vocab

    def convert2idx(self, words):
        """
        Convert `words` to indices. Use `unkWord` if not found.
        Optionally insert `bosWord` at the beginning and `eosWord` at the .
        """
        vec = []
        unk = self.lookup(self._UNK_WORD)
        vec += [self.lookup(word, default=unk) for word in words]

        return vec

    def convert2words(self, idx):
        """
        Convert `idx` to words.
        If index `stop` is reached, convert it and return.
        """
        words = []
        stop_idx = self.eos_idx
        for i in idx:
            if i == stop_idx:
                break

            w = self.get_word(i)
            if w is None:
                continue
            words += [w]

        return words


def load_vocabulary(vocab_file):
    vocab = Vocabulary()
    vocab.load_file(vocab_file)
    return vocab
