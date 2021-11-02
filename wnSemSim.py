# !pip install scipy.stats
# pip install spacy
import csv
import nltk
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

def preProcess(sentence):
    """
    Function for preprocessing a sentence. 
    The function tokenizes it, removes stopwords and punctuation.

    Params: 
        sentence: the sentence that will be preprocessed
    Returns:
        a list of preprocessed words that are parsed from the sentence
    """

    Stopwords = list(set(nltk.corpus.stopwords.words('english')))
    words = word_tokenize(sentence)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in Stopwords] #get rid of numbers and Stopwords
 
    return words

def wordSimilarity(w1,w2):
    """
    Function for determining the semantic similarity of individual words. 

    Params: 
        w1,w2: The words to compare
    Returns:
        The similarity score of words
    """

    S1 = ""
    S2 = ""
    
    if wn.synsets(w1):
        S1 = wn.synsets(w1)[0]
    if wn.synsets(w2):
        S2 = wn.synsets(w2)[0]
    
    if S1 and S2:
       similarity = S1.wup_similarity(S2)
       if similarity:
          return round(similarity,2)
    return 0

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
    
    tf = TfidfVectorizer(use_idf=True)
    tf.fit_transform([' '.join(words1), ' '.join(words2)])

    Idf = dict(zip(tf.get_feature_names(), tf.idf_))
    
    for w1 in words1:
        if w1 not in Idf:
            words1.remove(w1)
    for w2 in words2:
        if w2 not in Idf:
            words2.remove(w2)

    Sim = 0
    Sim_score1 = 0
    Sim_score2 = 0
    
    for w1 in words1:
        Max = 0
        for w2 in words2:
            score = wordSimilarity(w1,w2)
            if Max < score:
               Max = score
        Sim_score1 += Max*Idf[w1]
    Sim_score1 /= sum([Idf[w1] for w1 in words1])
    
    for w2 in words2:
        Max = 0
        for w1 in words1:
            score = wordSimilarity(w1,w2)
            if Max < score:
               Max = score
        Sim_score2 += Max*Idf[w2]
        
    Sim_score2 /= sum([Idf[w1] for w2 in words2])

    Sim = (Sim_score1+Sim_score2)/2
    
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
    
    pearsonCorrelation = pearsonr( wnSimilarity, humanSimilarity )[0]
    print("The pearson correlation between the human judgement and wordnet similarity is:")
    print(pearsonCorrelation)    

if __name__ == "__main__":
    main()
