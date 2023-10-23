# lovespacewhite.github.io

---

# [0-1.Command]

## Jupyter Notebook 명령어  
Shift + Enter : 셀실행 후, 아래셀 선택  
Alt + Enter : 셀실행 후, 아래 빈쉘 생성  
Ctrl + Enter : 셀실행  
A : 위쪽 빈쉘 생성  
B : 아래쪽 빈쉘 생성  
dd : 해당쉘 삭제  

---

# [0-2.도구 불러오기]

## pandas 불러오고, pd로 정의
~~~py
import pandas as pd """가나다"""
~~~

## numpy 불러오고, np로 정의
~~~py
import numpy as np
~~~

## seaborn 설치 및 불러오고, sns로 정의
(!: 리눅스 프롬프트 명령어)
~~~py
!pip install seaborn
import seaborn as sns 
~~~

## matplot 불러오고, plt로 정의
(%: 주피터랩 명령어)
~~~py
%matplotlib inline
import matplotlib.pyplot as plt
~~~

## 텐서플로 불러오고, tf로 정의
~~~py
import tensorflow as tf
~~~

## 텐서플로 케라스모델 및 기능 불러오기
(시퀀스(히든레이어개수)/덴스(노드개수)/액티베이션/과적합방지기능 불러오기)
~~~py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
~~~

## [모델] sklearn에서, 선형회귀모델(LinearRegression) 불러오기
~~~py
from sklearn.family import model
from sklearn.linear_model import LinearRegression
~~~

## [모델] sklearn에서, 분류회귀모델(Logistic Regression) 불러오기
(설명: 분류모델 주로 활용)
~~~py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  
~~~
 
## [모델] sklearn에서, 랜덤포레스트 불러오기
(설명: 의사결정나무 2개에서, 여러개를 더해 예측율을 높임)
~~~py
from sklearn.ensemble import RandomForestRegressor
~~~
 
## [모델] sklearn에서, 의사결정나무 불러오기
(설명: 분류/회귀가능한 다재다능, 다소복잡한 데이터셋도 학습가능)
~~~py
from sklearn.tree import DecisionTreeClassifier
~~~
 
## [모델] AdaBoost

## [모델] GBM (Gradient Boost)

## [모델] XGBoost
(설명: GBM의 느림, 과적합 방지를 위해 Regulation만 추가, 리소스를 적게 먹으며 조기종료 제공)

## [모델] SVM (Support Vector Machine)

## [모델] Auto Encoder

## [모델] CNN

## [모델] RNN

## [모델] LSTM

## [모델] Transformer

## [모델] SES (Simple Exponential Smoothing)

## [모델] YOLO

## [모델] VGG

---

# [1-1.빅데이터 수집]

## "00000.csv" 데이터 로드
(cp949는 MS office에서 인코딩할때 쓰임)
~~~py
df = pd.read_csv ("./00000.csv", encoding = "cp949")
~~~
 
## 커스텀 프레임웍에서 "00000.csv" 데이터 로드 2
(custom_framework.config.data_dir 폴더에서 불러옴)
~~~py
df = pd.read_csv (custom_framework.config.data_dir + "/00000.csv")
~~~
 
## 파일위치 환경변수
data 경로: custom_framework.config.data_dir  
workspace 경로: custom_framework.config.workspace_dir  
model 경로: custom_framework.config.model_dir  
log 경로: custom_framework.config.workspace_logs  

## "00000_final.csv" 데이터 저장 1
~~~py 
df.to_csv ("00000_final.csv", index = false)
~~~
 
## "00000_final.xlsx" 엑셀로 저장 2
~~~py
df.to_excel ("00000.xlsx")
~~~

---

# [1-2.빅데이터 분석]

Column Names = 열  
index = 행  
value = 값  

## 처음 위/아래 10개행을 보여주기
~~~py
df.head( )
df.head(10)
df.tail(10)
~~~

## df 데이터 형태(row/column수)
~~~py
df.shape
~~~

## df 데이터 컬럼내역 출력
~~~py
df.columns
~~~

## df 데이터 로우내역 출력
~~~py
df.values
~~~

## 자료구조 파악
~~~py
df.info( )
~~~

## 데이터타입 확인
~~~py
df.dtypes 
~~~

## 통계정보  
mean(평균), std(분산), min/max(최소/최대값)  
※ df.describe( ).transpose( )  
~~~py
df.describe( )
~~~

## 데이터 상관관계 분석
~~~py
df.corr( )
~~~

## x의 0번째 데이터 뽑아오기
~~~py
x[0]
x[-1] /*! x의 뒤에서 1번째 데이터 뽑아오기 */
~~~

## x의 0~4번째까지 데이터 뽑아오기
x[0:4]

## x의 전체 데이터 뽑아오기
x[:]

## 칼럼마다 결측치 여부 확인
df.isnull().sum()

## 00000 칼럼의 데이터 확인
df["00000"]

## 00000 칼럼의 값분포 확인
df["00000"].value_counts()

## 00000 칼럼의 값비율 확인
df["00000"].value_counts(normalize=True)

---

# [1-3.빅데이터 시각화]

---

# [1-4.빅데이터 전처리]

---

# [1-5.세트 구성]

## 트레이닝/테스트 세트 나누기
~~~py
from sklearn.model_selection import train_test_split 
~~~

## X,y데이터 불러오기

## X,y데이터 설정하기
~~~py
X = df.drop('Answer',axis=1).values
y = df['Answer'].values
~~~

## 테스트세트를 30%로 분류하고, 50번 랜덤하게 섞어 학습하기 
~~~py
X_train, X_test, y_test =
  train_test_split
  (X, y, test_size=0.30, random_state=50)
~~~

---

# [2.학습모델]

---

# [3.모델링]

---

# [4-1.최적화]

---

# [4-2.성능평가]

---

# [5.적용]

--------

-------

 

[1-3.빅데이터 시각화]

 

# 00000 칼럼 시각화 (이산)

df["00000"].value_counts().plot(kind="bar")

 

# 00000 칼럼 시각화 (연속)

df["00000"].plot(kind="hist")

 

# [Matplotlib] 시각화 사용법 (스캐터,바챠트)

## 영역 지정

plt.figure()

## 차트 및 값 지정

plt.plot()

## 시각화 출력

plt.show()

 

### 바 플롯

plt.bar(x, height)

### 히스토그램

plt.hist(x)

### 산점도

plt.scatter(x, y)

### 선 그래프

plt.plot(data)

 

## iris 데이터안의 variety 컬럼, 바차트 시각화 1

iris["variety"].value_counts( ).plot(kind="bar")

plt.show( )

 

## 시각화 2

df.corr( )["MonthlyCharges"][:-1].sort_values( ).plot(kind="bar")

sns.pairplot(df)

 

## iris 데이터안의 sepal.length 컬럼, 히스토그램 시각화 3

iris["sepal.length"].plot(kind="hist")

plt.show( )

 

 

# [Seaborn] 통계용 챠트 (히트맵)

## 카운트 플롯

sns.countplot(x="Churn", data=df)

## 박스 플롯

sns.boxplot(x="Churn", y="MonthlyCharges", data=df)

### 조인트 플롯

sns.jointplot(x="avg_bill", y="age", data=df, kind="hex")

### 상관관계 히트맵

sns.heatmap(df.corr( ), annot=True) 


-------

 

[1-4.빅데이터 전처리]

 

## 최고빈번값(Most frequent), 중앙값(Median), 평균값(Mean), 상수값(Constant)

 

## 입력데이터에서 제외

drop( ) 

## Null데이터 처리

dropna( ), fillna( ) 

## 누락데이터 처리

replace( )

※ axis=0(행), axis=1(열)

 

---

 

# 결측치

 

## 결측치 확인

### missing(결측값수)

 

## "_"를 numpy의 null값(결측치)으로 변경

df = df.replace("_", np.NaN)

 

## 결측치 행 제외시키기

### Listwise (행의 1개값이라도 NaN이면 제외)

df.dropna()

### Class 열의 결측치값 제외시키기

df.dropna(subset=["class"])

### Pairwise (행의 모든값이 NaN일때 제외)

df.dropna(how="all")

 

## 대체하여 채우기

### mean(평균) or median(중간)값 = 범주형데이터 주로사용

df.fillna(df.mean()["C1":"C2"])

### 앞값(ffill), 뒷값(backfill)

df = df.fillna(method="ffill")

### Most frequent(최빈값) = 범주형데이터 주로사용

 

### 선형방법 등 = 주변값과 상관관계로 선형 채우기

df = df.interpolate()

 

---

 

# 아웃라이어

 

## 아웃라이어 제외

 

### ex) Class열의 H값 제외후 변경

df = df [(df["class"]! = "H")]

 

## 아웃라이어 변경

 

### ex) Class열의 H값을 F값으로 변경

df["class"] = df["class"].replace("H", "F")


### ex) Q1, Q3, IQR 정의

Q1 = df[["Dividend","PBR"]].quantile(q=0.25)

Q3 = df[["Dividend","PBR"]].quantile(q=0.75)

IQR = Q3-Q1

 

#### IQR = Q3(3사분위수)-Q1(1사분위수)

#### 제거기준 = (Q3 + IQR * 1.5 보다 큰 값) & (Q1 - IQR * 1.5 보다 작은 값)

 

### 변경

IQR_df = df[(df["Dividend"] <= Q3["Dividend"]+1.5*IQR["Dividend"]) & (df["Dividend"] >= Q1["Dividend"]-1.5*IQR["Dividend"])]

IQR_df = IQR_df[(IQR_df["PBR"] <= Q3["PBR"]+1.5*IQR["PBR"]) & (IQR_df["PBR"] >= Q1["PBR"]-1.5*IQR["PBR"])]

IQR_df = IQR_df[["Dividend","PBR"]]

 

### 확인 (박스플롯)

IQR_df.boxplot()

IQR_df.hist(bins=20, figsize=(10,5))

 

---

 

# Feature Engineering

 

## 비닝(Binning)

## 연속형 변수를 범주형 변수로 만드는 방법

 

### cut = 구간값으로 나누기

q1 = df["avg_bill"].quantile(0.25)

q3 = df["avg_bill"].quantile(0.75)

 

df["bill_rating"] = pd.cut(

                 df["avg_bill"],

                 bins = [0, q1, q3, df["avg_bill"].max()],

                 labels = ["low", "mid", "high"])

print (df["bill_rating"].value_counts()]

 

### qcut = 구간개수로 나누기

df["bill_rating"] = pd.qcut(

                 df["avg_bill"],

                 3,

                 labels=["low", "mid", ;high"])

print (df["bill_rating"].value_counts()]

 

## 스케일링(Scaling)

### 데이터 단위크기를 맞춤으로서 표준화/정규화

 

#### Standard Scaling : 평균을 0, 표준편차를 1로 맞추기 (데이터 이상치가 심할경우 사용)

df_num = df[["avg_bill", "A_bill", "B_bill"]]

Standardization_df = (df_num - df_num.mean()) / df_num.std()

 

#### Min-Max Scaling : 모든 데이터를 0~1사이로 맞추기

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

nomalization_df = df_num.copy()

nomalization_df[:] = scaler.fit_transform(normalization_df[:])

 

-

 

# 원핫인코딩

 

## ex) 카테고리형 데이터를 원핫인코딩으로 컬럼 작성

cols = ["Gender", "Partner", "Dependents", "PhoneService",

 "MultipleLines", "InternetService", "OnlineSecurity",

 "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",

 "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]

 

dummies = pd.get_dummies(df[cols], drop_first=True)

df = df.drop(cols, axis=1)

df = pd.concat([df, dummies], axis=1)

 

## ex) 판다스로 쉽게 원핫인코딩

data = df[["AA","BB"]]

one_hot_df = pd.get_dummies(data, columns=["class"])

one_hot_df

 

 

# OrdinalEncoding

 

Categorical feature(범주형 특성)에 대한 순서형 코딩

각 범주들을 특성으로 변경하지 않고, 그 안에서 1,2,3 등의 숫자로 변경

범주가 너무 많아 one hot encoding을 하기 애매한 상황에서 이용

 

from category_encoders import OrdinalEncoder

enc1 = OrdinalEncoder(cols = "color")

df2 = enc1.fit_transform(df2)

df2

 

---

 

# 기타 주요작업

 

## ex) 토탈차지 공백을 0으로 변경후, 소수점 숫자형(float타입)으로 변경

df["TotalCharge"].replace([" "], ["0"], inplace=True)

df["TotalCharge"] = df["TotalCharge"].astype(float)

 

## ex) 해지여부 Yes/No를 1,0의 정수형으로 변경

df["Churn"].replace(["Yes", "No"], [1, 0], inplace=True)

 

## 카테고리데이터, 더미특성생성

get_dummies( ) 

 

## 새로운 뉴피처 추가

df["new_feature"] = df["f_1"]/df["f_2"]

 

## distinct(값종류수)

### distinct=1인 경우, 모든컬럼이 동일하므로 피처에서 제외

 

## 편향값

 

## 순서(인덱스)는 의미의 유무에 따라 제외

 

-------

 

[1-5.세트 구성]

 

## 트레이닝세트, 테스트세트 나누기

from sklearn.model_selection import train_test_split

 

-

 

## X, y 데이터 설정하기

## 00000 칼럼이 y값/타겟/레이블

X = df.drop("00000", axis=1).values

y = df["00000"].values

 

### ※ X, y 데이터 불러오기/만들기

### reshape(-1,1) 2차원배열 디자인포맷(reshape) 확장(-1은 알아서 넣으라는 뜻)

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).resharpe(-1,1)

y = np.array([13, 25, 34, 47, 59, 62, 79, 88, 90, 100])

 

-

 

## 테스트세트를 30%로 분류, 100번 랜덤하게 섞음

## y값이 골고루 분할되도록 stratify하게 분할

X_train, X_test, y_train, y_test = train_test_split(

                 X, y,

                 test_size=0.30, random_state=100, stratify = y)

 

-

 

## 데이터 정규화/스케일링

from sklearn.preprocessing import MinMaxScaler

help(MinMaxScaler)

 

scaler = MinMaxScaler( )

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

 

-

 

## ※ 참고

X_train.shape

[Out] (4930, 30)

y_train.shape

[Out] (4930,)

 

-------

 

[2.학습모델]

 

# LinearRegression 모델 (선형회귀)

 

## 모델 선정

model = LinearRegression( )

 

## 테스트 핏

model.fit(X_train, y_train)  

 

## 예측

predictions = model.predict(X_test)

 

## 확인

model.summary( )

 

# ※

hidden1 = model.layers[0]

weights, biase = hidden1.get_weights( )

print(weights)

 

---

 

# Logistic Regression 모델 (분류회귀)

 

## 라이브러리 불러오기

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report   ##성능지표 불러오기

 

## 데이터 불러오기

train = pd.read_csv(aidu_framework.config.data_dir + "/train.csv")

 

## 세트 나누기

X_train, X_test, y_train, y_test, = train_test_split(

                 train.drop("OOO", axis=1),   ## OOO(=트레이닝세트의 정답) 컬럼 삭제

                 train["OOO"], test_size=0.30, random_state=42)

 

## 모델링

model = LogisticRegression( )   ##불러올 모델 설정

model.fit(x_train, y_train)  ##테스트 핏

predictions = model.predict(x_test)  ##예측

print(classification_report(y_test, predictions)  ##성능지표 불러오기

 

---

 

# 의사결정나무(Decision Tree) (선형회귀)

## 분류/회귀가능한 다재다능, 다소복잡한 데이터셋도 학습가능

 

## 의사결정나무 라이브러리 불러오기

from sklearn.tree import DecisionTreeClassifier 

 

# ※

데이터셋(load_iris) 불러오기

from sklearn.datasets import load_iris 

 

iris = load_iris( )

X = iris.dat[ :, 2: ]   ##모든행에 대해서

y = iris.target

 

dtree_clf = DecisionTreeClassifier(max_depth=2)

dtree

 

---

 

# 딥러닝 모델

 

## 케라스 초기화

keras.backend.clear_session( )

 

## 모델 작성

model = Sequential( )

 

#### input layer

#### 30개의 features, 보통 연산효율을 위해 relu활용

model.add(Dense(64, activation="relu", input_shape=(30,)))

#### Batchnormalization을 활용

model.add(BatchNormalization( ))

#### 과적합 방지

model.add(dropout(0.5))

 

#### hidden layer 1

model.add(Dense(64, activation="relu"))

#### Batchnormalization을 활용

model.add(BatchNormalization( ))

#### 과적합 방지

model.add(dropout(0.5))

 

#### hidden layer 2

model.add(Dense(32, activation="relu"))

#### 과적합 방지

model.add(dropout(0.5))

 

#### output layer (아웃풋1(이진분류), sigmoid)

model.add(Dense(1, activation="sigmoid"))

#### ※ 또는 output layer (아웃풋3(다중분류), softmax)

model.add(Dense(3, activation="softmax"))

 

## 컴파일

### 이진분류 모델 (binary_crossentropy)

model.compile(optimizer="adam",

                       loss="binary_crossentropy",

                       metrics=["accuracy"])

### 다중분류 모델 (categorical_crossentropy) (원핫인코딩이 된 경우)

model.compile(optimizer="adam",

                       loss="categorical_crossentropy",

                       metrics=["accuracy"])

### 다중분류 모델 (sparse_categorical_crossentropy) (원핫인코딩이 안된 경우)

model.compile(optimizer="adam",

                       loss="sparse_categorical_crossentropy",

                       metrics=["accuracy"])

### 예측 모델

model.compile(optimizer="adam",

                       loss="mse")

 

## 조기종료 옵션

### 케라스 얼리스탑&체크포인트 불러오기

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

 

### 조기종료 : 로스값이 올라가면(5번까지는 괜찮음) 조기종료하기

early_stop = EarlyStopping(monitor="val_loss", mode="min",

                     verbose=1, patience=5)

 

### 체크포인트 : 최적 로스값을 기억(best_model.h5)하여 불러오기

check_point = ModelCheckpoint("best_model.h5", verbose=1,

                       monitor="val_loss", mode="min", save_best_only=True)

 

## 딥러닝 테스트 핏

model.fit(x=X_train, y=y_train,

        epochs=50, batch_size=20,

        validation_data=(X_test, y_test),

        verbose=1,

        callbacks=[early_stop, check_point])

 

## 학습과정 로그(loss,accuracy) history에 선언하여 남기기

history = model.fit(x=X_train, y=y_train,

        epochs=50, batch_size=20,

        validation_data=(X_test, y_test),

        verbose=1,

        callbacks=[early_stop, check_point])

 

## 학습로그 시각화

import matplotlib.pyplot as plt

 

plt.plot(history.history["accuracy"])

plt.plot(history.history["val_accuracy"])

plt.title("Accuracy")

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.legend(["train_acc", "val_acc"])

plt.show( )

 

---

 

# Ensemble 기법

## 1) Bagging

## 2) Boosting

이전학습 잘못예측한 데이터에 가중치부여해 오차보완

## 3) Stacking

여러개 모델이 예측한 결과데이터 기반, final_estimator모델로 종합 예측수행

## 4) Weighted Blending

각모델 예측값에 대해 weight 곱하여 최종 아웃풋계산

 

-

 

# XGBoost

!pip install xgboost

#### !는 리눅스 명령어

from xgboost import XGBClassfier

model = XGBClassifier(n_estimators=50)

model.fit(X_train,y_train)

pred = model.predict(X_test)

 

-

 

# LightGBM

!pip install lightGBM

from xgboost import GBMClassfier

model = LGBMClassifier(n_estimators=50)

model.fit(X_train,y_train)

pred = model.predict(X_test)

 

-

 

# 랜덤포레스트(Random Forest)

 - 선형회귀모델 중 하나

 - 의사결정나무 2개에서, 여러개를 더해 예측율을 높임

 

## 랜덤포레스트 불러오기

from sklearn.ensemble import RandomForestRegressor

 

## model을 랜덤포레스트로 정의하기

model = RandomForestRegressor

                 (n_estimators=50, ##학습시 생성할 트리갯수

                 max_depth=20, ##트리의 최대 깊이

                 random_state=42, ##난수 seed 설정

                 ...,

                 criterion="gini", ##분할 품질을 측정하는 기능(디폴트:gini)

                 min_samples_split=2, ##내부노드를 분할하는데 필요한 최소샘플수

                 min_samples_leaf=1, ##리프노드에 있어야할 최소샘플수

                 min_weight_fraction_leaf=0.0, ##가중치가 부여된 min_samples_leaf에서의 샘플수 비율

                 max_feature="auto") ##각노드에서 분할에 사용할 특징의 최대수

 

## 테스트 핏

model.fit(x_train, y_train)

 

## 스코어

model.score(x_test, y_test)

 

## 예측

prediction = model.predict(x_test)  

 

## RMSE값 구하기

np.mean((y_pred - y_test) ** 2) ** 0.5  

 

-------

 

[4-2.성능평가]

 

# 목표

## Loss(오차율) 낮추고, Accuracy(정확도) 높이기

## Error -> Epochs이 많아질수록 줄어들어야 함

## Epoch 많아질수록, 오히려 TestSet Error 올라가는경우 생길때, 직전Stop

## 학습시 조기종료(early stop) 적용되지 않았을 때는 개선여지가 있기에,

    배치사이즈나 에포크를 수정하여 개선할 수 있음

 

# 좋은모델

## 과적합(overfitting) : 선이 너무 복잡

### 트레인 어큐러시만 높아지고, 벨리드 어큐러시는 높아지지 않을때 (트레인어큐러시에 맞춰짐)

## 과소적합(underfitting) : 선이 너무 단순

### 트레인/벨리드 어큐러시가 교차되지 않고 아직 수평선을 향해 갈때

## 좋은모델 : 어느정도 따라가는 적당선

### 트레인/벨리드 어큐러시가 수평선을 이어 서로 교차될때

 

# 성능지표

## 오차행렬(Confusion Matrix) (분류모델에 주로 쓰임)

                 - TP (True Positive)

                 - TN (True Negative)

                 - FP (False Positive)

                 - FN (False Negative)

 

## 오차행렬 지표

                 - 정확도(Accuracy) = 맞춤(TP&TN) / 전체(total)

                 - 정밀도(Precision) = TP / TP + FP (예측한 클래스 중, 실제로 해당 클래스인 데이터 비율)

                 - 재현율(Recall) = TP = TP + FN (실제 클래스 중, 예측한 클래스와 일치한 데이터 비율)

                 - F1점수(F1-score) = 2 * [1/{(1/Precision)+(1/Recall)}] (Precision과 Recall의 조화평균)

                 - Support = 각 클래스 실제 데이터수

 

## 오차행렬 확인

import seaborn as sns

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, recall_score

 

y_true = [1, 0, 1, 1, 0, 1]

y_pred = [0, 0, 1, 1, 0, 0]

cm = confusion_matrix(y_true, y_pred)

cm

sns.heatmap(cm, annot=True)

precision_score(y_true, y_pred)

recall_score(y_true, y_pred)

 

## 오차행렬 성능지표 쉽게 확인하기

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

 

 

# 손실함수

## 회귀모델 손실함수(Loss Function)

                 - MSE(Mean Squared Error) : 실제에서 예측값 차이를 제곱, 합하여 평균 (예측)

                 - MAE(Mean Absolute Error) :

                 - CEE(Cross Entropy Error) : 예측결과가 빗나갈수록 더큰패널티 부여 (분류)

## 분류모델 손실함수

                 - Binary Cross Entropy (이진분류)

                 - Multi Class Classfication (다중분류)

 

## 주요 지표

                 - loss = MSE (학습시 사용한 loss function종류에 의해 결정) (작을수록 좋음)

                 - error = 실제값 빼기 예측값의 평균 (작을수록 좋음)

                 - MSE = 실제값 빼기 예측값 제곱의 평균 (작을수록 좋음)

                 - MAE = 실제값 빼기 예측값 절댓값의 평균 (작을수록 좋음)

                 - R2(결정계수) = 독립변수가 종속변수를 얼마나 잘설명하는지 (클수록 좋음)

 

# RMSE값 확인하기

## 회귀예측 주요 성과지표

import numpy as np

np.mean((y_pred - y_test) ** 2) ** 0.5

 

---

 

# 딥러닝 성능평가

losses = pd.DataFrame(model.history.history)

losses[["loss", "val_loss"]].plot( )

 

frome sklearn.metrics import classification_report, confusion_matrix

predictions = model.predict_classes(X_test)

print(classification_report(y_test, predictions))

print(confustion_matrix(y_test,predictions))

 

 

-------

 

[5.적용]

 

-------

 
