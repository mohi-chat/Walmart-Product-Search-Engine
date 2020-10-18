#!/usr/bin/env python
# coding: utf-8

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
import gensim
import os
from collections import OrderedDict
import math
from gensim.models import Word2Vec
import nltk
from numpy import dot
from numpy.linalg import norm
import gensim.downloader as api

class bm25:
    
    def __init__(self):
        
        # configuration options
        remove_stopwords = False  # or false
        use_stemming = False # or false
        remove_otherNoise = False # or false
        # global variables
        self.porter = PorterStemmer()

        # Generating documents
        # df1 = pd.read_csv("electronics.csv")
        # df2 = pd.read_csv("Personal Care.csv")
        # df3 = pd.read_csv("Food.csv")
        # df4 = pd.read_csv("Toys.csv")
        # df5 = pd.read_csv("Tools.csv")
        # df6 = pd.read_csv("Video Games.csv")
        # df7 = pd.read_csv("Clothings.csv")


        # frames = [df1, df2, df3, df4, df5, df6, df7]
        # self.df = pd.concat(frames)
        # self.df= self.df.drop_duplicates(subset="sku_id",keep="first")
        
        self.df=pd.read_csv('combined.csv')

        # Generate Vocab for tf-idf calculations and indexing
        D = []
        for index, row in self.df.iterrows():
            row['specifications'] = self.remove_punctuations(row['specifications'])
            val = row['title']+" "+row['specifications']+" "+str(row["price"])+" "+(row['category']+" ")*3
            val = val.lower();
            D.append(val)

        
        vocabulary = self.parser_tokenizer(D,remove_stopwords, use_stemming, remove_otherNoise)

        
        self.documentRepo = OrderedDict()

        # Creating dictionary for product id and title ,definitions

        for index, row in self.df.iterrows():
            row['specifications'] = self.remove_punctuations(row['specifications'])
            val = row['title']+" "+row['specifications']
            val = val.lower();
            self.documentRepo[row['sku_id']] = val


        self.indexDict = {}
        for word in vocabulary:
            s = set()
            for k,v in self.documentRepo.items():
                if word in v:
                    s.add(k)
                self.indexDict[word] = s

        info = api.info()  # show info about available models/datasets
        self.model_wiki = api.load("glove-wiki-gigaword-100")
        
        def load_gensim_model(model_path):
            model=gensim.models.KeyedVectors.load_word2vec_format(model_path)
            return model
        
        #self.model_wiki = load_gensim_model('word2vec.gz')

        self.avgdl = np.mean([len(self.documentRepo[docID].split()) for docID, v in self.documentRepo.items()])
    
    
    def remove_punctuations(self,doc):
        tokenizer = RegexpTokenizer(r'\w+')
        res = tokenizer.tokenize(doc)
        res = set(res)
        return " ".join(res)


    def query_expansion(self,query):
        query = query.split()
        query_words = [word for word in query if word not in stopwords.words('english')]
        new_query_words = []
        for i in query_words:
            if i.isalpha():
                res = self.model_wiki.most_similar(i)
                print(i,res[:5][0][0])
                new_query_words.append(res[:3][0][0])

        new_query_words = list(set(new_query_words))
        final_query_words = new_query_words + query_words
        return " ".join(final_query_words)


    def parser_tokenizer(self,data,remove_stopwords, use_stemming, remove_otherNoise):
        d = set()
        for line in data:
            for word in line.split():
                d.add(word)

        #  After removing stop words
        if remove_stopwords:
            d = [word for word in d if word not in stopwords.words('english')]
            print("Vocabulary size after removing stop words: ",len(d))

        # Stemming
        if use_stemming:
            stemmed_d = set()
            for word in d:
                stemmed_d.add(self.porter.stem(word))
            print("Vocabulary size after removing stop words and stemming: ",len(stemmed_d))
            d = stemmed_d

        # Remove other noise
        if remove_otherNoise:
            removeNoise_d = set()
            for word in d:
              # Removing characters other than letters.
                filtered_word = re.sub(r'[^A-Za-z0-9]', '', word)
                filtered_word = re.sub(r'[❖•]', '', filtered_word)
                if len(filtered_word) != 0:
                    removeNoise_d.add(filtered_word)
                    print("Vocabulary size after removing stop words, stemming and removing other noise: ",len(removeNoise_d))
                d = removeNoise_d
                
        return d


    def boolean_retrieval(self,query, return_all = True, verbose=True):
        query_words = query.split()
        matched_documents = set()
        for word in query_words:
            if word in self.indexDict:
                if(len(matched_documents)==0):
                    matched_documents = matched_documents | self.indexDict[word]
                else:
                    matched_documents = matched_documents | self.indexDict[word]
        matched_documents = set(matched_documents)
        if verbose:
            print("Query: ", query)
            print("\n")
            result = 1
            for d in matched_documents:
                print("Result ", result, ": ")
                print("Definition ID: ", d)
                print("Definition: ", self.documentRepo[d])
                print("\n")
                if (not return_all) and result == 5:
                    break
                result+=1

        return matched_documents


    def getTermFrequency(self,word, docWords):
        term_frequency = 0
        for d_word in docWords:
            if word in d_word:
                term_frequency+=1
        return term_frequency


    def calc_doc_BM25_score(self,query_words, docID):
        docWords = [word for word in self.documentRepo[docID].split()]
        N = len(self.documentRepo)
        D = len(docWords)
        k_1 = 1.2
        b = 0.75
        score = 0
        for word in query_words:
            if word in self.indexDict:
                tf_w = self.getTermFrequency(word, docWords)
                df_w = len(self.indexDict[word])
                IDF = math.log10(N/df_w)
                score = score + IDF*((tf_w * (k_1 + 1))/(tf_w + k_1 * (1 - b + b * (D/self.avgdl))))
        return score

    def BM25_rank_search(self,query, verbose = True):
        print(query)
        raw_search_result = self.boolean_retrieval(query,  return_all = True, verbose = False)
        query_words = query.split()
        docScores = {}
        for d in raw_search_result:
            docScores[d] = self.calc_doc_BM25_score(query_words, d) 
        top10Docs = self.printRankedResults(query, docScores, verbose)
        return top10Docs


    def printRankedResults(self,query, docScores, verbose = True):
        # Display ranked results
        printTopN = 10
        result = 1
        if verbose:
            print("Query: ", query)
            print("\n")
        top10Docs = []
        for docID, score in sorted(docScores.items(), key=lambda item: item[1], reverse = True):
            if printTopN > 0:
                if verbose: 
                    print("Result ", result, ": ")
                    print("Score ", score)
        #                 print("Entity: ", documentRepo[docID][0])
                    print("Definition ID: ", docID)
                    print("Definition: ", self.documentRepo[docID])
                    print("\n")
                top10Docs.append(docID)
                result+=1
                printTopN-=1
            else:
                break
        print(top10Docs)
        return top10Docs

    def generate_result_df(self,result):
        result_df = pd.DataFrame()
        for i in result:
            temp_df = self.df[self.df["sku_id"]==i]
            result_df = result_df.append(temp_df)
        return result_df

    # def load_gensim_model(model_path):
    #   model=gensim.models.KeyedVectors.load_word2vec_format(model_path)
    #   return model
    
    def search_query(self,query):
        result = self.BM25_rank_search(query,verbose=False)
        if(len(result)==0):
            print("in IF")
            query = self.query_expansion(query)
            query = self.remove_punctuations(query)
            result = self.BM25_rank_search(query)

        result_df = self.generate_result_df(result)
        return result_df


# In[10]:


def load_gensim_model(model_path):
    model=gensim.models.KeyedVectors.load_word2vec_format(model_path)
    return model


# In[11]:

def cos_similarity(a,b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim

def clean_query(query_str):
    q=query_str.lower()
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    q= tokenizer.tokenize(q)
    q=list(set(q))
    return q

def obtain_query_embeddings(q,model):
    v=np.zeros(100)
    l=0
    for i in q:
        if(i in model.vocab):
            temp=model.get_vector(i)
            v=v+temp
            l=l+1
    v=v/l
    return v

#clustering method
def clustering_method(query_str,processed_dataframe_path,model):

    def cos_similarity(a,b):
        cos_sim = dot(a, b)/(norm(a)*norm(b))
        return cos_sim

    def obtain_class_dict(data_frame1):
        class_dict={}
        for i in range(len(set(data_frame["labels"]))):
            embeds=data_frame1[data_frame1["labels"]==i]["vectors"].values
            class_dict[i]=np.mean(embeds,axis=0)
        return class_dict

    def clean_query(query_str):
        q=query_str.lower()
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        q= tokenizer.tokenize(q)
        q=list(set(q))
        return q

    def obtain_query_embeddings(q,model):
        v=np.zeros(100)
        l=0
        for i in q:
            if(i in model.vocab):
                temp=model.get_vector(i)
                v=v+temp
                l=l+1
        v=v/l
        return v

    def sort_by_similarity(c_dict,v,data_frame1):
        similarities=[cos_similarity(x,v) for x in c_dict.values()]
        pred=np.argmax(similarities)
        d_pred=data_frame1[data_frame1['labels']==pred]
        d_sim=[]
        #iterate over the dataframe of the predicted label
        for index,row in d_pred.iterrows():
            d_sim.append(cos_similarity(row["vectors"],v))
        d_pred["similarity"]=d_sim
        d_pred=d_pred.sort_values(by="similarity",ascending=False)
        return d_pred
    
    def query_label(c_dict,query_str,data_frame1,model):
        q=clean_query(query_str)
        v=obtain_query_embeddings(q,model)
        d_pred=sort_by_similarity(c_dict,v,data_frame1)
        return d_pred

    def query_label_no_sort(c_dict,query_str,data_frame1,model):
        q=clean_query(query_str)
        v=obtain_query_embeddings(q,model)
        similarities=[cos_similarity(x,v) for x in c_dict.values()]
        d_pred=data_frame1[data_frame1['labels']==np.argmax(similarities)]
        return d_pred

    
    data_frame=pd.read_pickle(processed_dataframe_path)
    c_dict=obtain_class_dict(data_frame)
    preds=query_label(c_dict,query_str,data_frame,model)
    return preds


# In[48]:


# var = bm25()
# res = var.search_query("apple ipad")


# In[56]:


# res = var.search_query(query)
# sim = query_title("apple ipad",var.model_wiki)


# In[60]:


def check_similarity(res, query_str,model):
    q=clean_query(query_str)
    v=obtain_query_embeddings(q,model)
    title = res['title'].values[0]
    title_tok = clean_query(title)
    title_vector = obtain_query_embeddings(title_tok,model)
    similarity=cos_similarity(v,title_vector)
    return similarity


    
    

