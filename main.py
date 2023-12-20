# Test code for IEEE course final project
# Fan Cheng, 2024

import minimatrix as mm

#Test your code here!
# The following code is only for your reference
# Please write the test code yourself

print("Test __str__")
test = mm.Matrix([[1],[2],[3]])
print(test)

print("\nTest shape")
test = mm.Matrix([[1,2,3]])
print(test.shape())

print("\nTest reshape")
test = mm.Matrix([[1,2,3]])
print(test.reshape((3,1)))
print(test.reshape((2,5)))

print("\nTest dot")
testa = mm.Matrix([[1],[2],[3]])
testb = mm.Matrix([[4,5,6]])
print(testa.dot(testb))

print("\nTest T")
test = mm.Matrix([[1,2,3],[4,5,6]])
print(test.T())

print("\nTest sum")
test = mm.Matrix([[1,2,3],[4,5,6]])
print(test.sum(axis=1))
print(test.sum(axis=0))
print(test.sum(axis=None))

print("\nTest Kronecker product")
testa = mm.Matrix([[1],[2],[3]])
testb = mm.Matrix([[4,5,6]])
print(testa.Kronecker_product(testb))

print("\nTest __getitem__")
test = mm.Matrix([[1,2,3],[4,5,6],[7,8,9]])
print(test[2,1])
print(test[1,1:3])
print(test[0:2,2])

print("\nTest __setitem__")
test = mm.Matrix([[1,2,3],[4,5,6],[7,8,9]])
test[2,1] = 0
print(test)
test[1,1:3] = mm.Matrix([[10,10]])
print(test)
test[0:2,2] = mm.Matrix([[20],[20]])
print(test)
test[1:3,1:3] = mm.Matrix([[30,30],[30,30]])
print(test)

print("\nTest __pow__")
test = mm.Matrix([[1,2,3],[4,5,6],[7,8,9]])
print(test**3)
print((test.dot(test)).dot(test))

print("\nTest __add__ , __sub__ , __mul__ , __len__")
testa = mm.Matrix([[1,2,3],[4,5,6],[7,8,9]])
testb = mm.Matrix([[9,8,7],[6,5,4],[3,2,1]])
print(testa + testb)
print(testa - testb)
print(testa * testb)
print(len(testa))

print("\nTest det")
testa = mm.Matrix([[1,2,9],[7,5,15],[4,8,2]])
print(testa.det())
testb = mm.Matrix([[0]])
print(testb.det())
testc = mm.Matrix([[0],[0],[0]])
print(testc.det())

print("\nTest inverse")
testa = mm.Matrix([[9,1,3],[5,1,2],[7,6,3]])
print(testa.inverse())
testb = mm.Matrix([[0,0,0],[0,0,0],[0,0,0]])
print(testb.inverse())

print("\nTest rank")
testa = mm.Matrix([[9,1,3],[5,1,2],[7,6,3]])
print(testa.rank())
testb = mm.Matrix([[0,0,0],[0,0,0],[0,0,0]])
print(testb.rank())

print("\nTest equation")
testa = mm.Matrix([[18,-12,0],[-12,28,-12],[0,-12,18]])
print(testa.inverse())
testb = mm.Matrix([[10],[0],[0]])
print(testa.equation(testb))

print("\nTest a series of initializing functions")
print(mm.I(3))
print(mm.narray((3,5),init_value=2))
print(mm.arange(2,8,2))
print(mm.zeros((3,4)))
print(mm.zeros_like(testa))
print(mm.ones((3,4)))
print(mm.ones_like(testa))
print(mm.nrandom((3,4)))
print(mm.nrandom_like(testa))

print("\nTest concatenate")
mat1 = mm.Matrix([[1, 2], [3, 4]])
mat2 = mm.Matrix([[5, 6], [7, 8]])
print(mm.concatenate([mat1,mat2], axis=0))
print(mm.concatenate([mat1,mat2], axis=1))

print("\nTest vectorize")
mat = mm.Matrix([[1, 2], [3, 4]])
print(mat)
v_func = mm.vectorize(lambda x: x * 2)
result = v_func(mat)
print(result)

print("\n最小二乘")
m = 1000
n = 100
X = mm.nrandom((m, n))
w = mm.nrandom((n, 1))
e = mm.nrandom((m, 1))

mean = e.sum()[0, 0] / m
e -= mm.narray((m, 1), mean)
Y = X.dot(w) + e
w_ = X.T().dot(X).inverse().dot(X.T()).dot(Y)

print(w.sum())
print(w_.sum())

@mm.vectorize
def square(x):
    return x ** 2

mse = square(w - w_).sum()[0, 0] / n
print(f"mse = {mse}")




