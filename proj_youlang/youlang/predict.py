#!/usr/bin/python
# -*- coding:utf-8 -*-

from sklearn.base import BaseEstimator, TransformerMixin

class Statskeywords(BaseEstimator, TransformerMixin):
    
    def __init__(self, topk = 100):
        self.topk = topk
#         print(self.topk)
        self.keywords = set()
        f = open("corpus/keywords.txt","r+", encoding='UTF-8')
        num = 0
        for content in f:
            if num < topk:
                self.keywords.add(content.strip().replace('\n', ''))
            num += 1
        f.close() 
        
        #初始化字典liwc
        self.liwc = {} 
        f2 = open("corpus/scliwc.txt",'r', encoding = 'gb18030')
        for ii in f2:     #ii在scliwc.txt中循环
            i = ii.strip().split() 
            self.liwc[i[0]] = i[1:len(i)]
        f2.close      
        
        self.category = set()
        for i in list(self.liwc.values()):
            for j in i:
                self.category.add(j)        
    
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X):
        '''
        文本中关键词的词频
        '''                        
        data = []
        for x in X:
            words = x.split()
            word_tf = []
            keycnt = 0
            for kw in self.keywords:
                word_tf.append(words.count(kw)) # 各个关键词的词频
                if kw in words:keycnt+=1
            word_tf.append(keycnt) # 关键词的个数
            
            psy = []
            for w in words:
                if w in self.liwc: #是否liwc字典包含分词结果列表words的哪些分词
                    psy += self.liwc[w]  
            cat_tf = []
            for cat in self.category:
                cat_tf.append(psy.count(cat))                
                
            data.append(word_tf + cat_tf)            
        return data        

class StatsFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.neg = set()
        f = open("corpus/neg_words.txt","r+", encoding='UTF-8')
        for content in f:
            self.neg.add(content)
        f.close()       

    def fit(self, X, y=None):
        return self

    def getcnt(self,x): 
        '''词个数'''
        return len(list(set(x.split())))

    def getnegcnt(self,x):
        '''负面词个数'''
        negcnt = 0
        words = x.split()
        for w in words:
            if w in self.neg:
                negcnt = negcnt+1
        return negcnt

    def getrepcnt(self,x):
        '''重复词个数'''
        repcnt =0
        words = x.split()        
        for w in list(set(words)):
            if words.count(w)>1: # 记录重复词汇（词频大于1）
                repcnt += 1
        return repcnt
    
    def transform(self, X):
        '''
        文本长度、词个数、词比例、
        负面词个数、负面词比例、
        重复词个数、重复词比例
        '''
        data = []
        for x in X:
            if len(x) == 0:
                length  = 1
            else :
                length = len(x)
            data.append([len(x),self.getcnt(x),self.getcnt(x)/length,
                         self.getnegcnt(x),self.getnegcnt(x)/length,
                         self.getrepcnt(x),self.getrepcnt(x)/length])            
        return data





















