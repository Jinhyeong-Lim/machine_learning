# Machine-Learing-


### Linear Regression (선형 회귀) 
##### 회귀식을 만들어 새로 들어온 데이터 값을 예측, 독립변수(원인)과 종속변수(결과)가 들어가는 식을 구하는 과정(연속형 종속변수)

![1](https://user-images.githubusercontent.com/64317686/114910809-4b726c00-9e59-11eb-95b2-bd8d7eb9e78a.JPG)

> - Linear(선형) : 직선으로 진행하는 모습을 가진 함수
> - Regression(회귀) : 예측, 분류하는 모델 fearture을 토대로 value 값 예측
> - 가설 h(x) = W*x + b  을 통해 실제 데이터와 함수와의 오차를 최소화 하는게 목적(최적의 기울기(w)와 b(절편)을 찾는게 목표)

##### Cost fuction(비용 함수) : 오차(Loss)에 대한 식, 함수의 값을 최소화 하는 목적을 가진 함수
> - Error(Loss) = h(x) - y
> - Square Error(Loss fuction) = (h(x)-y)^2
> - Mean Squared Error(Cost fuction) = (1/n) * Σ (h(x)-y)^2 (오차 제곱값들의 평균)
 
 
 ##### Optimizer(옵티마이저) : 경사 하강법(Gradient Descent)
 > ![1](https://user-images.githubusercontent.com/64317686/114913475-30edc200-9e5c-11eb-80f8-53cd6ef3013d.JPG)
 > - cost가 가장 작은 부분 볼록한 부분의 가장 아래 부분(기울기가 수렴하는 부분)의 매개 변수인 기울기(w) 와 절편(b)를 찾기 위한 작업
 
 >  ![1](https://user-images.githubusercontent.com/64317686/114914038-d3a64080-9e5c-11eb-8a5a-4e8ce280e6ca.JPG)
 >  ![2](https://user-images.githubusercontent.com/64317686/114914162-f33d6900-9e5c-11eb-8377-574614f9551a.JPG)
 > <br>
 > - 미분을 통해 cost를 최소화 하는 w 값을 업데이트, 접선의 기울기가 0이 될때(수렴할 때) 까지 반복한다.
 


### Logistic Regression (로지스틱 회귀)
##### 연속형 종속변수가 아닌 범주형 변수가 들어갈 때 쓰는 모델, 데이터가 특정 분류로 나뉘기 대문에 분류(Classification) 기법으로 볼 수 있다.

![1](https://user-images.githubusercontent.com/64317686/115115039-33235e00-9fcd-11eb-8963-ba163195a240.JPG)
> - 독립 변수 x가 [-∞,∞] 어느 숫자이든 상관 없이 종속 변수 y의 결과가 범위[0,1]로 제한된다. <br>
> - 가장 대표적인 sigmoid 함수는 활성화 함수(Activation Function)으로 사용 된다.

### KNN(k-nearest neighbor) 최근접 이웃 알고리즘
##### 새로운 데이터가 주어지면 주변(이웃) k(홀수) 개의 데이터를 살펴본 뒤 더 많은 데이터가 포함된 범주로 분류(Classification) 해준다.
![1](https://user-images.githubusercontent.com/64317686/115141837-628fa480-a079-11eb-92cc-d8aec4b49acd.JPG)
<br>
**장점** : 따로 훈련이 필요 없고(저사양 고성능) 노이즈 값 영향x, 데이터가 많으면 좋은 성능을 낸다. <br>
**단점** : 데이터가 고차원일 수록 성능 저하, k값에 따른 overfitting, underfitting 문제가 있다. -> k값 에 대한 연구가 필요하다, 새로운 데이터와 나머지 데이터 사이 거리 계산 시간이 오래걸린다.
