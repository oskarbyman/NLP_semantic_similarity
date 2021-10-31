# !pip install scipy.stats
import csv
import nltk
import numpy as np
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from scipy.stats import pearsonr

from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

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
    tokens = []
    raw_tokens = nltk.word_tokenize(sentence)
    for token in raw_tokens:
        if not token.lower() in nltk.corpus.stopwords.words("english"):
            for c in token:
                if not c.isalpha() or not c.isdigit():
                    tokens.append(token)
    tagged_tokens = list(dict.fromkeys(pos_tag(tokens)))
    return tagged_tokens

def generate_word_hypernyms(word):
    """
    Generate hypernyms for a word

    Params:
        word: a tagged word
    Returns:
        hypernyms: a set of hypernyms for the word
    """
    hypernyms = set()
    synsets = wn.synsets(word)
    for synset in synsets:
        hypernyms.update(set(synset.hypernyms()))
    return hypernyms

def generate_word_hyponyms(word):
    """
    Generate hyponyms for a word

    Params:
        word: a tagged word
    Returns:
        hyponyms: a set of hyponyms for the word
    """
    hyponyms = set()
    synsets = wn.synsets(word)
    for synset in synsets:
        hyponyms.update(set(synset.hyponyms()))
    return hyponyms

def process_word_pair(w1, w2):
    """
    Processes and calculates the intersection size divided by the unions size of a certain word pairs hypo- or hypernyms

    Params:
        w1: a tagged Word 1
        w2: a tagged Word 2
    Return:
        float number based on the result
    """
    if w1[1].startswith("NN"):
        w1_nyms = generate_word_hypernyms(w1[0])
        w2_nyms = generate_word_hypernyms(w2[0])
    elif w1[1].startswith("VB"):
        w1_nyms = generate_word_hyponyms(w1[0])
        w2_nyms = generate_word_hyponyms(w2[0])
    else:
        return 0
    
    nym_intersection = w1_nyms.intersection(w2_nyms)
    nym_union = w1_nyms.union(w2_nyms)
    try:
        result = len(nym_intersection) / len(nym_union)
    except ZeroDivisionError as zde:
        #print(f"nym_union was zero for {w1[0]} and {w2[0]}; {w1_nyms} and {w2_nyms}")
        return 0
    return result

def Sim(S1, S2):
    """
    Function for implementing the formula of Semantic Similarity presented in the project specification
    Calculates the max value of the 

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

    best_score = 0
    temp_result = 0
    for noun1 in S1_nouns:
        for noun2 in S2_nouns:
            temp_result = process_word_pair(noun1, noun2)
            if temp_result > best_score:
                best_score = temp_result
    noun_result = best_score

    best_score = 0
    for verb1 in S1_verbs:
        for verb2 in S2_verbs:
            temp_result = process_word_pair(verb1, verb2)
            if temp_result > best_score:
                best_score = temp_result
    verb_result = best_score

    return (noun_result + verb_result) / 2

def main():
    """
    Main function to run the test. Loads STSS-131 dataset from STSS-131-Dataset.csv file.
    """
    
    with open('STSS-131-Dataset.csv', newline='') as f:
        reader = csv.reader(f, delimiter=';')
        data = list(reader)
    
    wnHierSimilarity = []
    humanSimilarity = []
    
    for i in range(1,len(data)):
        S1 = data[i][1]
        S2 = data[i][2]
        humanSimilarity.append(float(data[i][3]))
        wnHierSimilarity.append(Sim(S1, S2))
    
    pearsonCorrelation = pearsonr(wnHierSimilarity, humanSimilarity)[0]
    print("The pearson correlation between the human judgement and hierarchical reasoning similarity is:")
    print(pearsonCorrelation)

if __name__ == "__main__":
    main()
