# !pip install scipy.stats
import csv
import nltk
import numpy as np
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from scipy.stats import pearsonr

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

"""
Script for measuring the semantic similarity of the STSS-131 dataset with the similarity formula defined in the project specifiactions
"""


def tokenize(S):
    """
    Tokenizes the sentence by splitting it into a list

    Params:
        S: the sentence as a string
    Returns:
        A list of tokens
    """
    return nltk.word_tokenize(S)

def tag(tokenized_S):
    """
    Function for tagging a tokenized sentence with part of speech tags

    Params:
        tokenized_S: a list of words in a sentence
    Returns:
        a list of tuples where each tuple contains the word and the tag associated with it, e.g. ('obtain', 'VB')
    """
    return pos_tag(tokenized_S)

def Sim(S1, S2):
    """
    Function for implementing the formula of Semantic Similarity presented in the project specification

    Params:
        S1: a string, Sentence 1
        S2: a string, Sentence 2
    Returns:
        similarity: The calculated similarity between the two input sentences
    """

    # Variables
    S1_nouns = []
    S1_verbs = []
    S2_nouns = []
    S2_verbs = []
    

def main():
    pass

if __name__ == "__main__":
    main()
