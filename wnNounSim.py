# !pip install scipy.stats
# pip install spacy
import csv
import nltk
from nltk.corpus.reader.wordnet import ADJ
import numpy as np
import spacy
from nltk.corpus import wordnet as wn
from scipy.stats import pearsonr

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')
"""
Script for measuring the semantic similarity of the STSS-131 dataset with WordNet Semantic Similarity in the style suggested in Lab 2
"""

def preProcess(S):
    """
    Function for preprocessing a sentence.
    The function tokenizes it, removes stopwords and punctuation.

    Params:
        sentence: the sentence that will be preprocessed
    Returns:
        a list of preprocessed words that are parsed from the sentence
    """

    nlp = spacy.load("en_core_web_sm")
    sentence = nlp(S)

    words = []
    for token in sentence:
        if not token.ent_type_ == "":
            continue
        elif token.is_stop:
            continue
        elif token.pos_ == "NOUN":
            words.append( token.text.lower() )
        elif token.pos_ == "VERB":
            a = convert( token.text, wn.VERB )
            if a != "":
                words.append( a )
        elif token.pos_ == "ADJ":
            a = convert( token.text, wn.ADJ )
            if a != "":
                words.append( a )
        elif token.pos_ == "ADV":
            a = convert( token.text, wn.ADV )
            if a != "":
                words.append( a )


    return words

def wordSimilarity(w1,w2):
    """
    Function for determining the semantic similarity of individual words.

    Params:
        w1,w2: The words to compare
    Returns:
        The similarity score of words
    """

    try:
        S1 = wn.synsets(w1)[0]
        S2 = wn.synsets(w2)[0]

        if S1 and S2:
            similarity = S1.wup_similarity(S2)
            if similarity:
                return round(similarity,2)
    except:
        return 0
    return 0

def convert(word, from_pos, to_pos=wn.NOUN):
    """ Transform words given from/to POS tags """

    synsets = wn.synsets(word, pos=from_pos)

    # Word not found
    if not synsets:
        return ""

    # Get all lemmas of the word (consider 'a'and 's' equivalent)
    lemmas = [l for s in synsets
                for l in s.lemmas()
                if s.name().split('.')[1] == from_pos
                    or from_pos in (wn.ADJ, wn.ADJ_SAT)
                        and s.name().split('.')[1] in (wn.ADJ, wn.ADJ_SAT)]

    # Get related forms
    derivationally_related_forms = [(l, l.derivationally_related_forms()) for l in lemmas]

    # filter only the desired pos (consider 'a' and 's' equivalent)
    related_noun_lemmas = [l for drf in derivationally_related_forms
                             for l in drf[1] 
                             if l.synset().name().split('.')[1] == to_pos
                                or to_pos in (wn.ADJ, wn.ADJ_SAT)
                                    and l.synset.name.split('.')[1] in (wn.ADJ, wn.ADJ_SAT)]

    # Extract the words from the lemmas
    words = [l.name for l in related_noun_lemmas]
    len_words = len(words)

    # Build the result in the form of a list containing tuples (word, probability)
    result = [(w, float(words.count(w))/len_words) for w in set(words)]
    result.sort(key=lambda w: -w[1])

    # return all the possibilities sorted by probability
    if not result:
        return ""
    return result[0][0]()


def Similarity(S1, S2):
    """
    Function for determining the semantic similarity of two senteces.
    First we remove the stopwords and punctuation and then compare individual words to the words on the other sentence.

    Params:
        S1,S2: The senteces to compare
    Returns:
        The similarity score of senteces
    """

    words1 = preProcess(S1)
    words2 = preProcess(S2)
    """
    tf = TfidfVectorizer(use_idf=True)
    tf.fit_transform([' '.join(words1), ' '.join(words2)])

    Idf = dict(zip(tf.get_feature_names(), tf.idf_))

    for w1 in words1:
        if w1 not in Idf:
            words1.remove(w1)
    for w2 in words2:
        if w2 not in Idf:
            words2.remove(w2)
    """
    Sim = 0
    count = 0

    for w1 in words1:
        for w2 in words2:
            Sim = Sim + wordSimilarity(w1,w2)
            Sim = Sim + wordSimilarity(w2,w1)
            count = count + 2

    if count > 0:
        Sim = Sim / count

    return round(Sim,2)

def main():
    """
    Main function to run the test. Loads STSS-131 dataset from STSS-131-Dataset.csv file.
    """

    with open('STSS-131-Dataset.csv', newline='', encoding="utf8") as f:
        reader = csv.reader(f, delimiter=';')
        data = list(reader)

    wnSimilarity = []
    humanSimilarity = []

    for i in range(1,len(data)):
        S1 = data[i][1]
        S2 = data[i][2]
        humanSimilarity.append( float( data[i][3] ) )
        wnSimilarity.append( Similarity(S1, S2) )

    #for i in range(len(humanSimilarity)):
    #    print(str(humanSimilarity[i]) + ", " + str(wnSimilarity[i]))

    pearsonCorrelation = pearsonr( wnSimilarity, humanSimilarity )[0]
    print("The pearson correlation between the human judgement and wordnet similarity is:")
    print(pearsonCorrelation)

if __name__ == "__main__":
    main()
