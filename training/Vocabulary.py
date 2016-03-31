import preprocess
import  itertools
from collections import Counter
import csv

class Vocabulary(object):
    def __init__(self):

        self.wordSet = set()
        self.vocabGrowth = 0

        sentences = []
        with open('data.csv', 'rb') as f:
            reader = csv.reader(f, delimiter=',')

            for row in reader:
                words = preprocess.clean(row[1])
                sentences.append(words)
                self.wordSet.update(words)

        padded_sentences = [ preprocess.pad(sentence) for sentence in sentences ]

        word_counts = Counter(itertools.chain(*padded_sentences))


        # Mapping from index to word
        self.vocabulary_inv = [x[0] for x in word_counts.most_common()]
        # Mapping from word to index
        self.vocabulary = {x: i for i, x in enumerate(self.vocabulary_inv)}

    def __len__(self):
        return len(self.wordSet)

    def addWord(self, word):
        word = word.encode('ascii', 'replace')
        if(word not in self.wordSet):
            self.vocabulary_inv.append(word)
            self.vocabulary[word] = self.vocabulary_inv.index(word)
            self.vocabGrowth += 1

    def getIdFromWord(self, word):
        word = word.encode('ascii', 'replace')
        return self.vocabulary[word]

    def getWordFromId(self, id):
        return self.vocabulary_inv[id]

    def getGrowth(self):
        return self.vocabGrowth

    def resetGrowth(self):
        self.vocabGrowth = 0
