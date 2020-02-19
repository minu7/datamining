from nltk.probability import FreqDist
from nltk.corpus import stopwords
import functools

def most_common_keywords(sentences, most_common):
    """
    Given the sentences returns N most common words

    Parameters
    ----------
    sentences: [[str]]
        Array of sentences lemmatized or stemmed

    most_common: int
        The number of most common words to return

    Raises
    ------
    RuntimeError

    Returns
    -------
        [str] Array with most common words ordered by frequency
    """
    words = functools.reduce(lambda x, y: x+y, sentences, [])
    # remove stop word for meaningful word frequencies
    a = stopwords.words('english')
    a = a + [',', '.', '$', '\'s', '%', '&', 'n\'t', '--', '’', 'would', 'also', '...', ';', '\'ll', 'u', 'ha', '\'ve', '\'re'];
    words = [x for x in words if x not in a]

    # count frequencies and get N most common words
    fdist = FreqDist(words)
    most_freq = fdist.most_common(most_common)

    most_freq = [x[0] for x in most_freq]
    return most_freq

def most_common_keywords_with_freq(sentences, most_common):
    """
    Given the sentences returns N most common words

    Parameters
    ----------
    sentences: [[str]]
        Array of sentences lemmatized or stemmed

    most_common: tuple
        The number of most common words to return with freq

    Raises
    ------
    RuntimeError

    Returns
    -------
        [str] Array with most common words ordered by frequency
    """
    words = functools.reduce(lambda x, y: x+y, sentences, [])
    # remove stop word for meaningful word frequencies
    a = stopwords.words('english')
    a = a + [',', '.', '$', '\'s', '%', '&', 'n\'t', '--', '’', 'would', 'also', '...', ';', '\'ll', 'u', 'ha', '\'ve', '\'re'];
    words = [x for x in words if x not in a]

    # count frequencies and get N most common words
    fdist = FreqDist(words)
    return fdist.most_common(most_common)


def words_presences(words, sentences):
    """
    Given the words to find in sentences returns an array with dim = [len(words), len(sentences)]
    which represent the presence of the corresponding word.
    Sentences and words must be preprocessed using stemming or lemmatization.


    Parameters
    ----------
    words: [str]
        Array of words

    sentences: [str]
        Array of sentences

    Raises
    ------
    RuntimeError

    Returns
    -------
        [[int]] Array with 0 or 1 representing words precences
    """
    presences = []
    for sentence in sentences:
        presences.append([1 if word in sentence else 0 for word in words])
    return presences

def words_counts(words, sentences):
    """
    Given the words to find in sentences returns an array with dim = [len(words), len(sentences)]
    which represent the number of words found in each sentence.
    Sentences and words must be preprocessed using stemming or lemmatization.

    Parameters
    ----------
    words: [str]
        Array of words

    sentences: [str]
        Array of sentences

    Raises
    ------
    RuntimeError

    Returns
    -------
        [[int]] Array of array with int representing words counts
    """
    counts = []
    for sentence in sentences:
        counts.append([sentence.count(word) for word in words])
    return counts
