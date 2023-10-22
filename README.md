#lovespacewhite.github.io

#[0-1.HW]

#[0-2.SW도구 불러오기]

##pandas 불러오고, pd로 정의하기 
~~~py
import pandas as pd
~~~

##numpy 불러오고, np로 정의하기
~~~py
import numpy as np
~~~

##seaborn 불러오고, sns로 정의하기
~~~py
!pip install seaborn
import seaborn as sns 
~~~

##matplot 불러오고, plt로 정의하기
~~~py
import matplotlib.pyplot as plt
%matplotlib inline
~~~

#[1-1.빅데이터 수집]

#[1-2.빅데이터 분석]

#[1-3.빅데이터 전처리]

#[1-4.세트 구성]

##트레이닝/테스트 세트 나누기
~~~py
from sklearn.model_selection import train_test_split 
~~~

##X,y데이터 불러오기

##X,y데이터 설정하기
~~~py
X = df.drop('Answer',axis=1).values
y = df['Answer'].values
~~~

##테스트세트를 30%로 분류하고, 50번 랜덤하게 섞어 학습하기 
~~~py
X_train, X_test, y_test =
  train_test_split
  (X, y, test_size=0.30, random_state=50)
~~~

#[2.학습모델]

#[3.모델링]

#[4-1.최적화]

#[4-2.성능평가]

#[5.적용]
