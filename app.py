from flask import Flask, request,redirect,url_for, render_template
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_val_score
from statistics import mean
from nltk.corpus import wordnet 
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from itertools import combinations
from time import time
from collections import Counter
import operator
#from xgboost import XGBClassifier
import math
import pickle
import openpyxl
from Treatment import diseaseDetail
from sklearn.linear_model import LogisticRegression
from flask import session
warnings.simplefilter("ignore")


app=Flask(__name__,static_url_path='/static')
app.secret_key = 'super secret key'    

global select_list

def synonyms(term):
    synonyms = []
    response = requests.get('https://www.thesaurus.com/browse/{}'.format(term))
    soup = BeautifulSoup(response.content,  "html.parser")
    try:
        container=soup.find('section', {'class': 'MainContentContainer'}) 
        row=container.find('div',{'class':'css-191l5o0-ClassicContentCard'})
        row = row.find_all('li')
        for x in row:
            synonyms.append(x.get_text())
    except:
        None
    for syn in wordnet.synsets(term):
        synonyms+=syn.lemma_names()
    return set(synonyms)

def similarity(dataset_symptoms,user_symptoms):
    found_symptoms = set()
    for idx, data_sym in enumerate(dataset_symptoms):
        data_sym_split=data_sym.split()
        for user_sym in user_symptoms:
            count=0
            for symp in data_sym_split:
                if symp in user_sym.split():
                    count+=1
            if count/len(data_sym_split)>0.5:
                found_symptoms.add(data_sym)
    found_symptoms = list(found_symptoms)
    return found_symptoms
 
def preprocess(user_symptoms):
    df_comb = pd.read_csv(r"\home\ubuntu\MediCURE-Disease-Prediction-based-on-Symptoms\Dataset\dis_sym_dataset_comb.csv") # Disease combination
    df_norm = pd.read_csv(r"\home\ubuntu\MediCURE-Disease-Prediction-based-on-Symptoms\Dataset\dis_sym_dataset_norm.csv") # Individual Disease

    X = df_comb.iloc[:, 1:] #symptoms
    Y = df_comb.iloc[:, 0:1] #diseases
    dataset_symptoms = list(X.columns)
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    splitter = RegexpTokenizer(r'\w+')
    processed_user_symptoms=[]
    for sym in user_symptoms:
        sym=sym.strip()
        sym=sym.replace('-',' ')
        sym=sym.replace("'",'')
        sym = ' '.join([lemmatizer.lemmatize(word) for word in splitter.tokenize(sym)])
        processed_user_symptoms.append(sym)
    user_symptoms = []

    for user_sym in processed_user_symptoms:
        user_sym = user_sym.split()
        str_sym = set()
        for comb in range(1, len(user_sym)+1):
            for subset in combinations(user_sym, comb):
                subset=' '.join(subset)
                subset = synonyms(subset) 
                str_sym.update(subset)
        str_sym.add(' '.join(user_sym))
        user_symptoms.append(' '.join(str_sym).replace('_',' '))
    return user_symptoms

@app.route("/",methods=["POST","GET"])
def index():
    return render_template("index.html")
@app.route("/about",methods=["POST","GET"])
def about():
    return render_template("about.html")
@app.route("/index",methods=["POST","GET"])
def index1():
    return render_template("index.html")
@app.route("/demo")
def demo():
    return render_template('demo.html')

@app.route("/predict",methods=["POST","GET"])
def predict():
    df_comb = pd.read_csv(r"\home\ubuntu\MediCURE-Disease-Prediction-based-on-Symptoms\Dataset\dis_sym_dataset_comb.csv") # Disease combination
    df_norm = pd.read_csv(r"\home\ubuntu\MediCURE-Disease-Prediction-based-on-Symptoms\Dataset\dis_sym_dataset_norm.csv") # Individual Disease
    X = df_comb.iloc[:, 1:] #symptoms
    Y = df_comb.iloc[:, 0:1] #diseases
    dataset_symptoms = list(X.columns)
    #print(dataset_symptoms)
    found_symptoms=set()    
    user_symptoms=list(request.form.get('symptoms','False').split(','))
    print(user_symptoms)
    user_symptoms= preprocess(user_symptoms)
    found_symptoms=similarity(dataset_symptoms,user_symptoms)
    print(found_symptoms)
    select_list=[]
    print("Top matching symptoms from your search!")
    for idx, symp in enumerate(found_symptoms):
        select_list.append(idx)
    dis_list = set()
    print(select_list)

    final_symp = [] 
    counter_list = []
    for idx in select_list:
        symp=found_symptoms[int(idx)]
        final_symp.append(symp)
        dis_list.update(set(df_norm[df_norm[symp]==1]['label_dis']))
    for dis in dis_list:
        row = df_norm.loc[df_norm['label_dis'] == dis].values.tolist()
        row[0].pop(0)
        for idx,val in enumerate(row[0]):
            if val!=0 and dataset_symptoms[idx] not in final_symp:
                counter_list.append(dataset_symptoms[idx])


    dict_symp = dict(Counter(counter_list))
    dict_symp_tup = sorted(dict_symp.items(), key=operator.itemgetter(1),reverse=True) 
    print("dictionary:",dict_symp_tup)
    another_symptoms=[]
    count=0
    for tup in dict_symp_tup:
        count+=1
        another_symptoms.append(tup[0])
    session['my_var'] = another_symptoms
    session['my_var2']=final_symp
    session['count']=count
    session['dict_symp_tup']=dict_symp_tup
    session['tup']=tup
    #session['dataset_symptoms']=dataset_symptoms


    return render_template("predict.html",found_symptoms=enumerate(found_symptoms),another_symptoms=enumerate(another_symptoms),count=count,dict_symp_tup=len(dict_symp_tup))

@app.route("/next",methods=["POST","GET"])
def next():
    #found_symptoms=[]

    my_var = session.get('my_var', None)
    my_var2=session.get('my_var2',None)
    count=session.get('count',None)
    #dataset_symptoms=session.get('dataset_symptoms',None)
    x=session.get('tup',None)
    df_comb = pd.read_csv(r"\home\ubuntu\MediCURE-Disease-Prediction-based-on-Symptoms\Dataset\dis_sym_dataset_comb.csv") # Disease combination
    df_norm = pd.read_csv(r"\home\ubuntu\MediCURE-Disease-Prediction-based-on-Symptoms\Dataset\dis_sym_dataset_norm.csv") # Individual Disease
    X = df_comb.iloc[:, 1:] #symptoms
    Y = df_comb.iloc[:, 0:1] #diseases
    dataset_symptoms = list(X.columns)
    final_symptoms=list(request.form.get('relevance','False').split(','))
    sample_x = [0 for x in range(0,len(dataset_symptoms))]
    for i in final_symptoms:
        my_var2.append(i)
        sample_x[dataset_symptoms.index(i)]=1
    
    session['sample_x']=sample_x
    print("sample_x: ",sample_x)

    
    return render_template("next.html",my_var2=enumerate(my_var2))

@app.route("/final",methods=["POST","GET"])
def final():
    sample_x=session.get('sample_x')
    print("samplexxxxxxxxx",sample_x)
    df_comb = pd.read_csv(r"\home\ubuntu\MediCURE-Disease-Prediction-based-on-Symptoms\Dataset\dis_sym_dataset_comb.csv") # Disease combination
    df_norm = pd.read_csv(r"\home\ubuntu\MediCURE-Disease-Prediction-based-on-Symptoms\Dataset\dis_sym_dataset_norm.csv") # Individual Disease
    #dataset_symptoms=session.get('dataset_symptoms',None)
    X = df_comb.iloc[:, 1:] #symptoms
    Y = df_comb.iloc[:, 0:1] #diseases
    dataset_symptoms = list(X.columns)
    my_var2=session.get('my_var2')
    print("final symptoms: ",my_var2)



    my_model=pickle.load(open(r'\home\ubuntu\MediCURE-Disease-Prediction-based-on-Symptoms\model_saved','rb'))
    #vectorizer=pickle.open(r'F:\SEM 6\Minor-2\Disease-Detection-based-on-Symptoms-master\model.pkl','rb')
    output=my_model.predict_proba([sample_x])
    scores = cross_val_score(my_model, X, Y, cv=10)

    k = 5
    diseases = list(set(Y['label_dis']))
    diseases.sort()
    topk = output[0].argsort()[-k:][::-1]
    print(f"\nTop {k} diseases predicted based on symptoms")
    topk_dict = {}
    # Show top 10 highly probable disease to the user.
    for idx,t in  enumerate(topk):
        match_sym=set()
        row = df_norm.loc[df_norm['label_dis'] == diseases[t]].values.tolist()
        row[0].pop(0)

        for idx,val in enumerate(row[0]):
            if val!=0:
                match_sym.add(dataset_symptoms[idx])
        prob = (len(match_sym.intersection(set(my_var2)))+1)/(len(set(my_var2))+1)
        prob *= mean(scores)
        topk_dict[t] = prob
    #j = 0
    topk_index_mapping = {}
    topk_sorted = dict(sorted(topk_dict.items(), key=lambda kv: kv[1], reverse=True))
    print("this is top_k_sorted",topk_sorted)
    # print("tok_sorted_first_key",list(topk_sorted.items()[0][0]))
    # prob1=list(topk_sorted.items()[0][1])
    # prob2=list(topk_sorted.items()[1][1])
    arr=[]
    #i=0

    for key in topk_sorted:
        prob = topk_sorted[key]*100
        arr.append(f' Disease name: {diseases[key]}\t Probability: {str(round(prob, 2))}%')
        #arr[i]=diseases[key]
        #i=i+1
    
    return render_template("final.html",arr=arr)
@app.route("/treatment",methods=["POST","GET"])
def treatment():
    treat_dis=request.form.get('dis','False').lower()
    workbook = openpyxl.load_workbook('F:\SEM 6\Minor-2\Disease-Detection-based-on-Symptoms-master\cure minor.xlsx')
    worksheet = workbook['Sheet1']
    #arr2=[]
    ans=[]
    for row in worksheet.iter_rows(values_only=True):
        print("rowwwwwwww",row)
        print("disssssssss",treat_dis)
        if treat_dis in row:
            print("seccodd",treat_dis)
            stri = ''.join(row[1:])
            ans=stri.split(',')
            print("answer:",ans)
    return render_template("treatment.html",ans=ans)



if __name__=='__main__':
    app.run(debug=True, host="0.0.0.0", port=5000,threaded=True)