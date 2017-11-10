from __future__ import print_function
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys
import mlp
import svm
import rf
import logreg
import libspacy3
import numpy as np
#reload(sys)
#sys.setdefaultencoding('utf8')
def main():
  test_file = 'evaluation.csv'
  train_file = 'train.csv'
  print("Beginning INDIX")

  #Read the train_file
  X_raw=[]
  y_raw=[]
  df = pd.read_csv(train_file)
  df = df.dropna(axis=1, how='all')
  df=df.fillna(0)
  print("Generating features")
  for (url,aa,bc,label) in tqdm(zip(df['url'],df['additionalAttributes'],df['breadcrumbs'],df['label'])):
    X = generate_features(str(url), str(aa), str(bc))
    y = label_to_int(label)
    X_raw.append(X)
    y_raw.append(y)
    if(len(y_raw) > 1000):break


  num_train=int(0.8*len(y_raw))
  
  X_raw_test=[]
  y_raw_test=[]
  tdf = pd.read_csv(test_file)
  tdf = tdf.dropna(axis=1, how='all')
  tdf=tdf.fillna(0)
  print("Generating features")
  for (url,aa,bc,labelid) in tqdm(zip(tdf['url'],tdf['additionalAttributes'],tdf['breadcrumbs'],tdf['id'])):
    X_test_data = generate_features(str(url), str(aa), str(bc))
    y_raw_test.append(int(labelid))
    X_raw_test.append(X_test_data)
    if(len(y_raw_test) > 1000):break

  X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.20, random_state=42)
  y_pred,y_eva_log = logreg.fit_predict(X_train, y_train, X_test, y_test, X_raw_test)
  y_pred,y_eva_mlp = mlp.fit_predict(X_train, y_train, X_test, y_test, X_raw_test)
  y_pred,y_eva_svm = svm.fit_predict(X_train, y_train, X_test, y_test, X_raw_test)
  y_pred,y_eva_rf = rf. fit_predict(X_train, y_train, X_test, y_test, X_raw_test)
  fp = open('submission.csv','w')
  for i in range(len(y_raw_test)):
    fp.write(str(y_raw_test[i])+", "+str(y_eva_log[i])+", "+str(y_eva_mlp[i])+", "+str(y_eva_svm[i])+","+str(y_eva_rf[i])+"\n")
  fp.close()


def label_to_int(label):
  labels={'books':0,'music':1,'videos':2,'rest':3}
  return labels[label]

def generate_features(url, aa, bc):
  vec = libspacy3.get_vector(bc)
  aavec = libspacy3.get_vector(aa)
  vec = np.concatenate((vec,aavec),axis=0)
  return vec


if __name__=="__main__":
  main()
