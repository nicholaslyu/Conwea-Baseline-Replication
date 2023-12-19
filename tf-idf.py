coarse_nyt = "data/nyt/coarse/df.pkl"
coarse_20news = "data/20news/coarse/df.pkl"
fine_nyt = "data/nyt/fine/df.pkl"
fine_20news = "data/20news/fine/df.pkl"
coarse_nyt_seedword = "data/nyt/coarse/seedwords.json"
coarse_20news_seedword = "data/20news/coarse/seedwords.json"
fine_nyt_seedword = "data/nyt/fine/seedwords.json"
fine_20news_seedword =  "data/20news/fine/seedwords.json"



def read_seedwords(path):
    import json
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    return data

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
import numpy as np
df_20news_c = pickle.load(open(coarse_20news, "rb"))
df_20news_fine = pickle.load(open(fine_20news, "rb"))
df_nyt_c = pickle.load(open(coarse_nyt,"rb"))
df_nyt_fine = pickle.load(open(fine_nyt, "rb")) 
nyt_c_seedwords = read_seedwords(coarse_nyt_seedword)
nyt_fine_seedwords= read_seedwords(fine_nyt_seedword)
twtnews_c_seedwords = read_seedwords(coarse_20news_seedword)
twtnews_fine_seedwords = read_seedwords(fine_20news_seedword)


class TF_IDF_Classifier:
    def __init__(self, data,seedwords,type_,dataset):
        self.vectorizer = TfidfVectorizer()
        self.data = data
        self.X = self.vectorizer.fit_transform(self.data["sentence"])
        self.labels = self.data["label"]
        self.dataset=dataset
        self.type_ = type_
        self.seedwords = seedwords
        self.prediction = []

    def predict(self,impute):
        tf_idf_all_doc = []
        predicted_labels= []
        for i in range(len(self.labels)):
            tf_idf_by_class = {}
            for k,v in self.seedwords.items():
                current_class = 0 
                for word in v:
                    word_index = self.vectorizer.vocabulary_.get(word)
                    if word_index is not None:
                        current_class +=self.X[i,word_index]
                tf_idf_by_class[k] = current_class/len(v)
    ##return the key with the max value (tf-idf)
            predicted_class = max(tf_idf_by_class, key=tf_idf_by_class.get)
            if tf_idf_by_class[predicted_class] == 0 and self.type_=="coarse":
                predicted_labels.append(impute)
            else:
                predicted_labels.append(predicted_class) 
        return predicted_labels
    
    def compute_f1(self,pred):
        macro = f1_score(self.labels, pred , average='macro')
        micro = f1_score(self.labels, pred , average='micro')
        output = self.dataset + " " + self.type_ +" "
        output = output + "macro F1 " + str(macro) +" " + "micro F1 " +str(micro)
        print(output)
        return output




if __name__ == "__main__":
    output_results = ""
    
    #NYT fine grained
    a = TF_IDF_Classifier(df_nyt_fine,nyt_fine_seedwords,"fine","nyt")
    result = a.predict("")
    output = a.compute_f1(result)
    output_results+=output
    output_results+="\n"
    
    #NYT Coarse
    b = TF_IDF_Classifier(df_nyt_c,nyt_c_seedwords,"coarse","nyt")
    for mis in nyt_c_seedwords.keys():
        result = b.predict(mis)
        print("impute with "+mis)
        output = b.compute_f1(result) + "impute with " + mis +"\n"
        output_results+=output
    output_results+="\n"
        
    #20news fine grained
    c = TF_IDF_Classifier(df_20news_fine,twtnews_fine_seedwords,"fine","20news")
    result = c.predict("")
    output =c.compute_f1(result)
    output_results+=output
    output_results+="\n"
    
    #20news coarse
    d = TF_IDF_Classifier(df_20news_c,twtnews_c_seedwords,"coarse","20news")
    for mis in twtnews_c_seedwords.keys():
        result = d.predict(mis)
        print("impute with "+mis)
        output = d.compute_f1(result) + "impute with " + mis +"\n"
        output_results+=output
        
    f = open("tf_idf_results.txt", "w")
    f.write(output_results)
    f.close()
    
    print("all results saved!")
        
    
        
        
    
        