import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import pickle
import json
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
from gensim.models import KeyedVectors
from gensim import models
import pandas as pd
#read seed words


def read_seedwords(path):
    import json
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    return data
def cosine_similarity(v1,v2):
    return dot(v1, v2)/(norm(v1)*norm(v2))

class W2V_Classifer:
    def __init__(self, tokenized, vector_size, window,epochs,dataset,type_):
        self.seedwords = read_seedwords(dataset_paths[dataset]['seedwords'][type_])
        self.model = Word2Vec(sentences=tokenized, 
              vector_size=vector_size,
              window=window, 
              workers=8,
              sg=0,
              epochs = epochs
              )
        self.dataset = dataset
        self.type_ = type_
        self.data = tokenized
        self.parameters = {"window_size":window,"vector_size":vector_size,"epochs":epochs}
        
    def generate_label_vector_agg(self,seed_words):
        labels_agg = {}
        for label, seeds in seed_words.items():
            labels_agg[label] = np.mean([self.model.wv[word] for word in seeds if word in self.model.mv], axis=0)
        return labels_agg

            
        
    def predict(self):
        seed_words = dataset_paths[self.dataset]["seedwords"][self.type_]
        seed_words = read_seedwords(seed_words)
        label_representations = self.generate_label_vector_agg(self.seedwords)
        
        predictions = []
        for doc in self.data:
            doc_representation = np.mean([self.model.wv[word] for word in doc if word in self.model.wv], axis=0)
            cos_sim_by_class = {}
            for label,representation in label_representations.items():
                cos_sim_by_class[label] = cosine_similarity(doc_representation, representation)
            predicted_class = max(cos_sim_by_class, key=cos_sim_by_class.get)
            predictions.append(predicted_class)
        return predictions
    
    def compute_f1(self,true,pred):
        macro = f1_score(true, pred , average='macro')
        micro = f1_score(true, pred , average='micro')
        output = self.dataset + " " + self.type_ + " with parameters " + str(self.parameters) + "\n"
        output = output + "macro F1 " + str(macro) +" " + "micro F1 " +str(micro)
        print(output)
        return output
        
    
    
    

if __name__ == "__main__":
    
    dataset_paths = {"nyt":{"data":{"fine":"data/nyt/fine/df.pkl",
                                        "coarse":"data/nyt/coarse/df.pkl"},
                                "seedwords":{"coarse":"data/nyt/coarse/seedwords.json",
                                            "fine":"data/nyt/fine/seedwords.json"}},
                         "20news":{"data":{"fine":"data/20news/fine/df.pkl",
                                        "coarse":"data/20news/coarse/df.pkl"},
                                    "seedwords":{"coarse":"data/20news/coarse/seedwords.json",
                                            "fine":"data/20news/fine/seedwords.json"}}}
    
    nltk.download('punkt')
    coarse_nyt = "data/nyt/coarse/df.pkl"
    coarse_20news = "data/20news/coarse/df.pkl"
    fine_nyt = "data/nyt/fine/df.pkl"
    fine_20news = "data/20news/fine/df.pkl"
    coarse_nyt_seedword = "data/nyt/coarse/seedwords.json"
    coarse_20news_seedword = "data/20news/coarse/seedwords.json"
    fine_nyt_seedword = "data/nyt/fine/seedwords.json"
    fine_20news_seedword =  "data/20news/fine/seedwords.json"
    
    
    outputstring = ""
    #20news coarse
    windows = [5,10,15,20,25]
    df_20news_c = pickle.load(open(coarse_20news, "rb"))
    data = df_20news_c["sentence"]
    labels = df_20news_c["label"]
    tokenized = [word_tokenize(doc.lower()) for doc in data]
    for w in windows:
        test = W2V_Classifer(tokenized,300,w,5,"20news","coarse")
        predictions = test.predict()
        outputstring+=test.compute_f1(labels,predictions)

    #20news fine grained
    df_20news_fine = pickle.load(open(fine_20news, "rb"))
    data = df_20news_fine["sentence"]
    labels = df_20news_fine["label"]
    tokenized = [word_tokenize(doc.lower()) for doc in data]
    for w in windows:
        test = W2V_Classifer(tokenized,300,w,5,"20news","fine")
        predictions = test.predict()
        outputstring+=test.compute_f1(labels,predictions)

    #nyt coarse
    df_nyt_coarse = pickle.load(open(coarse_nyt, "rb"))
    data = df_nyt_coarse["sentence"]
    labels = df_nyt_coarse["label"]
    tokenized = [word_tokenize(doc.lower()) for doc in data]
    for w in windows:
        test = W2V_Classifer(tokenized,300,w,5,"nyt","coarse")
        predictions = test.predict()
        outputstring+=test.compute_f1(labels,predictions)

    #nyt fine-grained
    df_nyt_fine = pickle.load(open(fine_nyt, "rb"))  
    data = df_nyt_fine["sentence"]
    labels = df_nyt_fine["label"]
    tokenized = [word_tokenize(doc.lower()) for doc in data]
    for w in windows:
        test = W2V_Classifer(tokenized,300,w,5,"nyt","fine")
        predictions = test.predict()
        outputstring+=test.compute_f1(labels,predictions)
        
        
    f = open("w2v_results.txt", "w")
    f.write(outputstring)
    f.close()
    



    
        
    
    
    
