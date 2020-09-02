import numpy as np
a = np.array([0.3, 2.9, 4.0])

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/ sum_exp_a
    return y

def softmax2(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/ sum_exp_a
    return y



y = softmax(a)
y2 = softmax2(a)
print(y)
print(np.sum(y))
print('\n', y2)
print(np.sum(y2))
