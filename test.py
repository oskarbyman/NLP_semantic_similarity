# !pip install scipy.stats
import csv
import nltk
import numpy as np
from nltk.corpus import wordnet as wn
from scipy.stats import pearsonr

from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')

def wup(S1, S2):
    return S1.wup_similarity(S2)

def preProcess(sentence):
    Stopwords = list(set(nltk.corpus.stopwords.words('english')))
    words = word_tokenize(sentence)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in Stopwords] #get rid of numbers and Stopwords
 
    return words

def wordSimilarity(w1,w2,num):
    print(w1)
    
    S1 = ""
    S2 = ""
    
    if wn.synsets(w1):
        S1 = wn.synsets(w1)[0]
    if wn.synsets(w2):
        S2 = wn.synsets(w2)[0]
    
    if S1 and S2:
       similarity = wup(S1, S2)
       if similarity:
          return round(similarity,2)
    return 0
    
def Similarity(T1, T2, num):
    words1 = preProcess(T1)
    words2 = preProcess(T2)
    
    tf = TfidfVectorizer(use_idf=True)
    tf.fit_transform([' '.join(words1), ' '.join(words2)])

    Idf = dict(zip(tf.get_feature_names(), tf.idf_))
    
    Sim = 0
    Sim_score1 = 0
    Sim_score2 = 0
    
    for w1 in words1:
        Max = 0
        for w2 in words2:
            score = wordSimilarity(w1,w2,num)
            if Max < score:
               Max = score
        Sim_score1 += Max*Idf[w1]
    Sim_score1 /= sum([Idf[w1] for w1 in words1])
    
    print(round(Sim_score1,2))
    for w2 in words2:
        Max = 0
        for w1 in words1:
            score = wordSimilarity(w1,w2,num)
            if Max < score:
               Max = score
        print(w2)
        Sim_score2 += Max*Idf[w2]
        
    Sim_score2 /= sum([Idf[w1] for w2 in words2])
    print(round(Sim_score2,2))

    Sim = (Sim_score1+Sim_score2)/2
    
    return round(Sim,2)

def main():
    
    with open('STSS-131-Dataset.csv', newline='') as f:
        reader = csv.reader(f, delimiter=';')
        data = list(reader)
    
    wnSimilarity = []
    humanSimilarity = []
    
    
    
    S1 = data[22][1]
    S2 = data[22][2]
    humanSimilarity.append(data[22][3])
    wnSimilarity.append( Similarity(S1, S2, 0) )
    
    
    for i in range(1,len(data)):
        S1 = data[i][1]
        S2 = data[i][2]
        humanSimilarity.append(data[i][3])
        wnSimilarity.append( Similarity(S1, S2, 0) )
    
    
    
    

if __name__ == "__main__":
    main()