__author__ = 'Nick'
import pandas as pd
import numpy as np
import string

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer,TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem.snowball import RussianStemmer
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import Normalizer,OneHotEncoder
from sklearn.decomposition import TruncatedSVD
import nltk
from sklearn.cross_validation import  KFold,cross_val_score
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score
import scipy.sparse as sp
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib
import time
import os

normalizer = RussianStemmer()
table = str.maketrans({key: None for key in string.punctuation+string.digits})

def readSW(fn):
  #returns a list of stop words read from file fn, separeted by new line
  return  [line.rstrip('\n') for line in open(fn,"r")]

def mynormalize(doc):
  #returns normalzied list of words using global normalizer
  return (normalizer.stem(w) for w in doc)

def tokenize(text):
  #returns normalized and tokenized words
  return mynormalize(nltk.word_tokenize(text))

def preprocess(text):
  #preprocess the text deleting punctuation and digits according to global variable table
  return str(text).translate(table).lower()

class ItemSelector(BaseEstimator, TransformerMixin):
  #class for heterogeneous data selecting based on key value (column)
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

class PriceExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, docs):
      d = [{'price': float(i)}
        for i in np.nditer(docs)]
      return d

class FeaturesGetter(BaseEstimator, TransformerMixin):
  #transform pandas data frame to numpy array for feature processing
    def fit(self, x, y=None):
        return self

    def transform(self, docs):
        features = np.recarray(shape=(len(docs),),
#                              dtype=[('description', object), ('title', object)])
                               dtype=[('description', object), ('title', object),('price',object)])
        i=0
        for index,row in docs.iterrows():
          features['description'][i] = row['description']
          features['title'][i] = row['title']
          features['price'][i] = row['price']
          i=i+1
        return features

def makeMask(lvl,dic_categories):
  #return mask array for hierarchy level lvl based on dic_categories
  #assume hierarchy level difference is 1
  res=[None]*len(dic_categories.items())
  vec_s = sorted(dic_categories.items(), key=lambda x: x[1])
  prevtxt = ''
  cls=0
  for i,(k,v) in enumerate(vec_s):
    txts = v.split('|')
    if(len(txts)<=lvl):
      txt = txts[lvl-1]
    else:
      txt = txts[lvl]
    if(prevtxt!=txt and i>0):
      cls+=1
    res[i]=cls
    prevtxt = txt
  return res

def reindex(arr,index):
  #return reindexed arrary arr based on array index
  r=[None]*len(arr)
  for i,val in enumerate(arr):
    r[i]=index[arr[i]]
  return r

def getMaxLvl(texts):
  #function to get max hierarchy level based on count of separator
  m=0
  for k,t in texts.items():
    c = str(t).count('|')
    if(c>=m):
      m=c
  return m

def main():
  myStopWords = readSW("..\\..\\mini_data\\sw.txt")
  modelfold = "..\\model_all"
  modelfn = modelfold + "\\cls.pkl"
  df_categories = pd.read_csv("..\\..\\mini_data\\category_sorted.csv", header = 0,delimiter = ",")
  df = pd.read_csv("..\\..\\data\\train.csv", header = 0,delimiter = ",")
  # df = pd.read_csv("..\\..\\mini_data\\avg_train.csv", header = 0,delimiter = ",")
  df_test_mini = pd.read_csv("..\\..\\mini_data\\train.csv", header = 0,delimiter = ",")#subset of train.csv (first 1000+ examples)
  df_test = pd.read_csv("..\\..\\data\\test.csv", header = 0,delimiter = ",")

  dic_categories = df_categories.set_index('category_id')['name'].to_dict()

  combined_features = FeatureUnion(
    transformer_list=[
      ('description_tf_idf',Pipeline([
        ('selector',ItemSelector(key='description')),
        ('tfidf',TfidfVectorizer(analyzer='word',preprocessor=preprocess,tokenizer=tokenize, stop_words=myStopWords,max_df=0.7)),
      ])),
      ('title_bin',Pipeline([
        ('selector',ItemSelector(key='title')),
        ('vec_bin',TfidfVectorizer(analyzer='word',preprocessor=preprocess,tokenizer=tokenize,binary=True)),
      ])),

      ('price',Pipeline([
        ('selector',ItemSelector(key='price')),
        ("price_norm",Normalizer()),
        ('price',PriceExtractor()),
        ('vec',DictVectorizer()),
    ])),
      ],
     transformer_weights={
       'title_bin':1,
       'description_tf_idf':0.8,
      'price':0.5
     },
     n_jobs=-1,
    )

  #vectorizing using hashing
  combined_features_hashing = FeatureUnion(
    transformer_list=[
      ('description_tf_idf',Pipeline([
        ('selector',ItemSelector(key='description')),
        ('tf_hashing',HashingVectorizer(analyzer='word',preprocessor=preprocess,tokenizer=tokenize, stop_words=myStopWords,non_negative=True,n_features=2 ** 18)),
        ('tfidf',TfidfTransformer()),

      ])),
      ('title_bin',Pipeline([
        ('selector',ItemSelector(key='title')),
        ('vec_bin',HashingVectorizer(analyzer='word',preprocessor=preprocess,tokenizer=tokenize,binary=True,non_negative=True)),
        ('tfidf',TfidfTransformer()),
      ])),

      ('price',Pipeline([
        ('selector',ItemSelector(key='price')),
        ("price_norm",Normalizer()),
        ('price',PriceExtractor()),
        ('vec',DictVectorizer()),
    ])),
      ],
     transformer_weights={
       'title_bin':1,
       'description_tf_idf':0.8,
      'price':0.5
     },
     n_jobs=-1,
    )


  pipeline = Pipeline([
  ('getFeatures',FeaturesGetter()),
   ('features',combined_features_hashing),
   # ('features',combined_features),
  # ('classifierNB',MultinomialNB()),
  ('classifierSVM',SVC(kernel='linear'))
 ])

  end_learning = 0;start_learning = 0
  if(not os.path.exists(modelfold)):
    os.mkdir(modelfold)
  if(not os.path.exists(modelfn)):
    start_learning = time.time()
    print("learning step")
    pipeline.fit(df,df['category_id'].values)
    end_learning = time.time()

    print ("saving classifier model")
    joblib.dump(pipeline,modelfn)

  else:
    print ("loading classifier model")
    pipeline = joblib.load(modelfn)

  start_predicting= time.time()
  print("classification step")
  # predictions = pipeline.predict(df_test)
  end_predicting= time.time()

  #write results to file
  # f = open("result.csv","w")
  # f.write("item_id,category_id\n")
  # for item, c in zip(df_test["item_id"].values,predictions):
  #   f.write(str(item)+','+str(c)+'\n')
  # f.close()

  print("timeLearning %d , timePredicting %d" % (end_learning  - start_learning, end_predicting-start_predicting ))
  # print("kFold step")
  # score = cross_val_score(pipeline,df,df['category_id'].values,n_jobs=-1,scoring='accuracy')
  # print("kFold accuracy avg: %f" % (sum(score)/len(score)))

  print("hierarchy accuracy calculation for one file")
  acc=dict()
  maxLvl = getMaxLvl(dic_categories)
  # acc[maxLvl] = accuracy_score(df_test_mini['category_id'].values ,predictions)
  # for lvl in range(maxLvl):
  #   classes = makeMask(lvl,dic_categories)
  #   h_predictions = reindex(predictions,classes)
  #   h_categories = reindex(df_test_mini['category_id'].values,classes)
  #   acc[lvl] = accuracy_score(h_categories ,h_predictions )
  #   print("accuracy:%f,lvl: %d" % (acc[lvl],lvl))
  # print("accuracy: %f,lvl: %d" % (acc[maxLvl],maxLvl))




  print("start cross validation for hierarchy accuracy ")
  k_fold = KFold(n=len(df),n_folds=3)
  for learn_ind, test_ind in k_fold:

    print("pipline fitting")
    pipeline.fit(df.iloc[learn_ind],df.iloc[learn_ind]['category_id'].values)
    print("pipline predicting")
    predictions = pipeline.predict( df.iloc[test_ind])

    for lvl in range(maxLvl):
      classes = makeMask(lvl,dic_categories)
      h_predictions = reindex(predictions,classes)
      h_categories = reindex(df.iloc[test_ind]['category_id'].values,classes)
      acc[lvl] = accuracy_score(h_categories ,h_predictions )
    #print results
    for key,val in acc.items():
      print("accuracy:%f,lvl: %d" % (val,key))

  #some parameters' tuning
  # parameters = {'features__description_tf_idf__tfidf__use_idf':(True,False),
  #               'features__description_tf_idf__tfidf__max_df':(0.7,0.3,0.9),
  #               'features__title_bin__vec_bin__binary':(False,True),}

  #gs_cls = GridSearchCV(pipeline,parameters,n_jobs=-1)
  #gs_cls =gs_cls.fit(df,df['category_id'].values)
  #print (score)
  # best_parameters, score, _ = max(gs_cls.grid_scores_, key=lambda x: x[1])
  # for param_name in sorted(parameters.keys()):
  #   print("%s: %r" % (param_name, best_parameters[param_name]))



if __name__ == '__main__':
  main()
