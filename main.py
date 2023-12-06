# Test code for IEEE course final project
# Fan Cheng, 2024

import minimatrix1 as mm

#Test your code here!
# The following code is only for your reference
# Please write the test code yourself

mat_a = mm.Matrix([[0,3.0],[2.0,0]])
print(mat_a)








'''mat = mm.Matrix([[1,2,3],[6,5,4],[7,8,9]])
print(mat)
print(mat.data)
print(mat.dim)
print(mat.inverse())
print(mat.rank())
print(mat.shape())
print(mat.T)
print(mat.det())'''

# m24 = mm.arange(0, 24)
'''print(m24)
print(m24.reshape([3, 8]))
print(m24.reshape([4, 6]))
'''
'''print(mm.zeros_like(m24))
print(mm.ones_like(m24))
m2 = mm.nrandom((3, 3))
print(m2)
print(mm.nrandom_like(m24))
'''

"""m = 1000
n = 100
X = mm.nrandom((m, n))
w = mm.nrandom((n, 1))
e = mm.nrandom((m, 1))


mean = e.sum()[0, 0] / m
e -= mm.narray((m, 1), mean)
Y = X.dot(w) + e
w_ = X.T().dot(X).inverse().dot(X.T()).dot(Y)"""


'''print()
print(w.sum())
print(w_.sum())'''

"""@mm.vectorize
def square(x):
    return x ** 2

mse = square(w - w_).sum()[0, 0] / n
print(mse)"""


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



