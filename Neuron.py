class Neuron:
    # 가중치 절편 초기화
    def __init__(self):
        self.w = 1.0
        self.b = 1.0

    # 정방향 계산 만들기
    def forpass(self, x):
        y_hat = x * self.w + self.b  # 직선 방정식 계산
        return y_hat

    # 역방향 계산 만들기
    def backprop(self, x, err):
        w_grad = x * err  # 가중치에 대한 Gradient를 계산
        b_grad = 1 * err  # 절편에 대한 Gradient를 계산
        return w_grad, b_grad

    # 훈련 메서드 만들기
    def fit(self, x, y, epochs=10000):
        for i in range(epochs):
            for x_i, y_i in zip(x, y):
                y_hat = self.forpass(x_i)
                err = -(y_i - y_hat)
                w_grad, b_grad = self.backprop(x_i, err)
                self.w -= w_grad
                self.b -= b_grad


from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt

diabetes = load_diabetes()
x = diabetes.data[:,2]
y = diabetes.target # 배열 슬라이싱 샘플 0 1 2 뽑아내기

print(x.shape, y.shape)

neuron = Neuron()
neuron.fit(x, y)

plt.scatter(x, y)
pt1 = (-0.1, -0.1 * neuron.w + neuron.b)
pt2 = (0.15, 0.15 * neuron.w + neuron.b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
plt.xlabel('x')
plt.ylabel('y')
plt.show()
