from Neural_Network import NN

structure = [2, 4, 1]
nn = NN(structure)
x = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
y = [[1], [0], [0], [1]]

w, b = nn.train_lm(x, y, 0.1, 1000, 0.001)
nn.test(x, y)
print('W: \n', w)
print('B: \n', b)
print(nn.predict([[0, 0], [0, 1], [1, 0], [1, 1]]))
