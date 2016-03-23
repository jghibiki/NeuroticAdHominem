import itertools
from collections import Counter
import csv

# Globals
vocabulary_inv = None
vocabulary = None

from options import Options
options = Options()
Options = options


from models.TextCNN import TextCNN

import training
from training.train import train
from training import preprocess

from hosting.host import launch as hosting_launch
from hosting.host import eval

from app.Server import launch as app_launch

def launch():
    hosting_launch()
    app_launch()
    print(eval("bill clinton is an idiot"))


sentences = []
with open('data.csv', 'rb') as f:
    reader = csv.reader(f, delimiter=',')

    for row in reader:
        words = preprocess.clean(row[1])
        sentences.append(words)

padded_sentences = [ preprocess.pad(sentence) for sentence in sentences ]

word_counts = Counter(itertools.chain(*padded_sentences))

# Mapping from index to word
vocabulary_inv = [x[0] for x in word_counts.most_common()]
# Mapping from word to index
vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

del word_counts
del padded_sentences
del row
del words
del sentences
