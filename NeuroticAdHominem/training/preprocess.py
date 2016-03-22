
from NeuroticAdHominem import Options as opts

urlFinder = re.compile('\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*')
atNameFinder = re.compile(r'@([A-Za-z0-9_]+)')
atNameCounter = 0

exclude_punc = set([
        "!",
        "?",
        ".",
        ",",
        ":",
        ";",
        "'",
        "\"",
        "“",
        "’",
        "-"
])

def clean(string):
    words = []

    for word in string \
        .strip() \
        .replace("&amp;", "") \
        .replace("&gt;","") \
        .replace("&lt;", "") \
        .lower().split():

        if urlFinder.match(word):
            words.append("<URL/>")
        elif atNameFinder.search(word):
            words.append("<AT_NAME/>")
            atNameCounter +=1
        else:
            word = ''.join(ch for ch in word if ch not in exclude_punc)
            words.append(word)

def pad(sentence):

    if(ops.sentence_padding > len(words)):
        raise Error("Increase sentence_padding, found sentence that is %s words long. sentence_padding must be greater than or equal to the number of words in the longest sentence" % len(words))
    else:
        for x in range(opts.sentence_padding - len(words)):
            words.append(opts.sentence_padding_token)


