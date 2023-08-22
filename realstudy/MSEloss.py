import tensorflow as tf

w = tf.Variable(2.)

def f(w):
  z = 2*w**2 + 5
  return z

with tf.GradientTape() as tape:
  z = f(w)

gradients = tape.gradient(z, [w])
print(gradients)

w = tf.Variable(4.0)    #임의 선언
b = tf.Variable(1.0)

def hypothesis(x):
  return w*x + b

def mse_loss(y_pred, y):
  # 두 개의 차이값을 제곱을 해서 평균을 취한다.
  return tf.reduce_mean(tf.square(y_pred - y))

x = [3, 5, 7, 9, 11, 13, 15, 17, 19] 
y = [30, 49, 68, 89, 107, 127, 145, 164, 182] # 맵핑되는 값

optimizer = tf.optimizers.SGD(0.0001) #경사하강법 with lr = 0.01로 지정

for i in range(101):
  with tf.GradientTape() as tape:
    # 현재 파라미터에 기반한 입력 x에 대한 예측값을 y_pred
    y_pred = hypothesis(x)

    # 평균 제곱 오차를 계산
    cost = mse_loss(y_pred, y)

  # 손실 함수에 대한 파라미터의 미분값 계산
  gradients = tape.gradient(cost, [w, b])

  # 파라미터 업데이트
  optimizer.apply_gradients(zip(gradients, [w, b]))

  if i % 10 == 0:
    print("epoch : {:3} | w의 값 : {:5.4f} | b의 값 : {:5.4} | cost : {:5.6f}".format(i, w.numpy(), b.numpy(), cost))

print(hypothesis(x))