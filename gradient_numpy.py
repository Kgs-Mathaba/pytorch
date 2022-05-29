import numpy as np

# f = w*x
# f = 2*x
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0

# model prediction
def forward(x):
    return w * x


# loss = MSE
def loss(y, y_predicted):
    return ((y - y_predicted) ** 2).mean()


# gradients for
# MSE = 1/N *(w*x - y)**2
# dJ/dw = 1/2N *(w*x - y)**2
def gradients(x, y, y_predicted):
    return np.dot(2 * x, y_predicted - y).mean()


print(f"Prediction before training: f(5) = {forward(5):.3f}")


# training
learning_rate = 0.01
n_iter = 20
for epoch in range(n_iter):
    # prediction = foraward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients
    dw = gradients(X, Y, y_pred)

    # update weights
    w = w - learning_rate * dw
    if epoch % 2 == 0:
        print(f"epoch: {epoch}, loss: {l:.3f}, w: {w:.3f}")

print(f"Prediction after training: f(5) = {forward(5):.3f}")

