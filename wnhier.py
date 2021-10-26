# !pip install scipy.stats
import csv
import nltk
import numpy as np
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from scipy.stats import pearsonr

from nltk.tokenize import word_tokenize
nltk.download('stopwords')

"""
Script for measuring the semantic similarity of the STSS-131 dataset with the similarity formula defined in the project specifiactions
"""


def generate_synset(word):
    """
    Function for generating the nltk synset for a certain word

    Params: 
        word: the target word for the synset
    Returns:
        A wn synset of the word
    """
    return wn.synsets(word)

def preprocess(sentence):
    """
    Function for preprocessing a sentence. 
    The function tokenizes it and tags it with the appropriate POS tags
    Removes stopwords and punctuation.

    Params: 
        sentence: the sentence that will be preprocessed
    Returns:
        a list of preprocessed words that are parsed from the sentence
    """
    _tokens = nltk.word_tokenize(sentence)
    for token in raw_tokens:
        if not token.lower() in nltk.corpus.stopwords.words("english"):
            if not c.isalpha() for c in token:
                if not c.isdigit() for c in token:
                    tokens.append(token)
    tagged_tokens = pos_tag(tokens)
    return tagged_tokens

def generate_sentence_hypernyms(sentence):
    """
    Generate hypernyms for every word in a sentence
    Set is used to prevent duplicate hypernyms

    Params:
        sentence: a tokenized and tagged list of words
    Returns:
        a set of hypernyms for the sentence
    """
    hypernyms = set()
    for word in sentence:
        word = word[0]
        synsets = wn.synsets(word)
        for sysnset in synsets:
            hypernyms.update(set(synset.hypernyms()))
    return hypernyms

def generate_sentence_hyponyms(sentence):
    """
    Generate hyponyms for every word in a sentence
    Set is used to prevent duplicate hyponyms

    Params:
        sentence: a tokenized and tagged list of words
    Returns:
        a set of hyponyms for the sentence
    """
    hyponyms = set()
    for word in sentence:
        word = word[0]
        synsets = wn.synsets(word)
        for sysnset in synsets:
            hyponyms.update(set(synset.hyponyms()))
    return hyponyms

def Sim(S1, S2):
    """
    Function for implementing the formula of Semantic Similarity presented in the project specification

    Params:
        S1: a string, Sentence 1
        S2: a string, Sentence 2
    Returns:
        similarity: The calculated similarity between the two input sentences
    """

    S1_preprocessed = preprocess(S1)
    S2_preprocessed = preprocess(S2)
    
    S1_nouns = [token for token in S1_preprocessed if token[1].startswith("NN")]
    S1_verbs = [token for token in S1_preprocessed if token[1].startswith("VB")]
    S2_nouns = [token for token in S2_preprocessed if token[1].startswith("NN")]
    S2_verbs = [token for token in S2_preprocessed if token[1].startswith("VB")]
    
    # Create hypernyms for all nouns in both sentences
    S1_noun_hypernyms = generate_sentence_hypernyms(S1_nouns)
    S2_noun_hypernyms = generate_sentence_hypernyms(S2_nouns)

    # Create hyponyms for all verbs in both sentences
    S1_verb_hyponyms = generate_sentence_hyponyms(S1_verbs)
    S2_verb_hyponyms = generate_sentence_hyponyms(S2_verbs)

    # Create the unions and intersections used in the formula
    verb_hyponyms_int = S1_verb_hyponyms.intersection(S2_verb_hyponyms)
    verb_hyponyms_uni = S1_verb_hyponyms.union(S2_verb_hyponyms)

    noun_hypernyms_int = S1_noun_hypernyms.intersection(S2_noun_hypernyms)
    noun_hypernyms_uni = S1_noun_hypernyms.union(S2_noun_hypernyms)

    try:
        noun_result = len(noun_hypernyms_int) / len(noun_hypernyms_uni)
        verb_result = len(verb_hyponyms_int) / len(verb_hyponyms_uni)
    except Error as e:
        print(f"Error occured: {e}")
        pass
    
    similarity = 

def main():
    pass

if __name__ == "__main__":
    main()
