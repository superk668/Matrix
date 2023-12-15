# Test code for IEEE course final project
# Fan Cheng, 2024

import minimatrix1 as mm

#Test your code here!
# The following code is only for your reference
# Please write the test code yourself

#Test __str__
test = mm.Matrix([[1],[2],[3]])
print(test)

#Test shape
test = mm.Matrix([[1,2,3]])
print(test.shape())

#Test reshape
test = mm.Matrix([[1,2,3]])
print(test.reshape((3,1)))
print(test.reshape((2,5)))

#Test dot
testa = mm.Matrix([[1],[2],[3]])
testb = mm.Matrix([[4,5,6]])
print(testa.dot(testb))

#Test T
test = mm.Matrix([[1,2,3],[4,5,6]])
print(test.T())

#Test sum
test = mm.Matrix([[1,2,3],[4,5,6]])
print(test.sum(axis=1))
print(test.sum(axis=0))
print(test.sum(axis=None))

#Test Kronecker product
testa = mm.Matrix([[1],[2],[3]])
testb = mm.Matrix([[4,5,6]])
print(testa.Kronecker_product(testb))

#Test __getitem__
test = [[1,2,3],[4,5,6],[7,8,9]]
print(test[2][1])
print(test[1][1:3])
"""print(test[0:2][2])""" #error here

#Test __setitem__



#Test __pow__
test = mm.Matrix([[1,2,3],[4,5,6],[7,8,9]])
print(test**3)
print((test.dot(test)).dot(test))

#Test __add__ , __sub__ , __mul__ , __len__
testa = mm.Matrix([[1,2,3],[4,5,6],[7,8,9]])
testb = mm.Matrix([[9,8,7],[6,5,4],[3,2,1]])
print(testa + testb)
print(testa - testb)
print(testa * testb)
print(len(testa))

#Test det
testa = mm.Matrix([[1,2,9],[7,5,15],[4,8,2]])
print(testa.det())
testb = mm.Matrix([[0]])
print(testb.det())
testc = mm.Matrix([[0],[0],[0]])
print(testc.det())

#Test inverse


#Test rank

#Test I(n)
print(mm.I(3))
print(mm.narray((3,5),init_value=2))
print(mm.arange(2,8,2))
print(mm.zeros((3,4)))
print(mm.zeros_like(testa))
print(mm.ones((3,4)))
print(mm.ones_like(testa))
print(mm.nrandom((3,4)))
print(mm.nrandom_like(testa))




m = 1000
n = 100
X = mm.nrandom((m, n))
w = mm.nrandom((n, 1))
e = mm.nrandom((m, 1))


mean = e.sum()[0, 0] / m
e -= mm.narray((m, 1), mean)
Y = X.dot(w) + e
w_ = X.T().dot(X).inverse().dot(X.T()).dot(Y)


print()
print(w.sum())
print(w_.sum())

@mm.vectorize
def square(x):
    return x ** 2

mse = square(w - w_).sum()[0, 0] / n
print(mse)

"""a = mm.Matrix([[1,0,0],[0,1,0],[0,0,1]])
print(a.rank())"""



'''@vectorize
def func(x):
    return x ** 2
x = Matrix([[1, 2, 3],[2, 3, 1]])

print(func(x))'''

'''@vectorize
def my_abs(x):
    if x < 0:
        return -x
    else:
        return x
y = Matrix([[-1, 1], [2, -2]])
print(my_abs(y))'''



