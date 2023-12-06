import random

class Matrix:
    def __init__(self, data=None, dim=None, init_value=0):
        if data:
            self.data = data
            row = len(data)
            column = len(data[0])
            for r in range(row):
                if len(data[r]) != column:
                    print("这不是一个矩阵！")
            self.dim = (row, column)
        elif dim:
            self.dim = dim
            self.data = [[init_value for c in range(dim[1])] for r in range(dim[0])]
        else:
            print("你没有提供任何数据！")
    
    
    def shape(self):
        return self.dim
    
    def reshape(self, newdim):
        if self.dim[0] * self.dim[1] == newdim[0] * newdim[1]:
            data_lst = [0] * (self.dim[0] * self.dim[1])
            for r in range(self.dim[0]):
                for c in range(self.dim[1]):
                    data_lst[r * self.dim[1] + c] = self.data[r][c]
            ans_lst = [data_lst[r * newdim[1] : (r + 1) * newdim[1]] for r in range(newdim[0])]
            return Matrix(ans_lst)
        else:
            print("维数错误")
            return Matrix([[0]])

    def dot(self, other):
        if self.dim[1] != other.dim[0]:
            print("These 2 matrices has no point product.")
            return Matrix([[0]])
        lst_mat = [[0 for c in range(other.dim[1])] for r in range(self.dim[0])]
        for r in range(self.dim[0]):
            for c in range(other.dim[1]):
                for i in range(self.dim[1]):
                    lst_mat[r][c] += self[r, i] * other[i, c]
        ans_mat = Matrix(lst_mat)
        return ans_mat
    
    def T(self):
        T_mat = Matrix(dim = (self.dim[1], self.dim[0]))
        for r in range(T_mat.dim[0]):
            for c in range(T_mat.dim[1]):
                T_mat.data[r][c] = self.data[c][r]
        return T_mat

    def sum(self, axis=None):
        if axis == 0:
            return Matrix([[sum([self.data[i][j] for i in range(self.dim[0])]) for j in range(self.dim[1])]])
        if axis == 1:
            return Matrix([[sum([self.data[i][j] for j in range(self.dim[1])])] for i in range(self.dim[0])])
        elif not axis:
            return Matrix([[sum(self.data[i][j] for i in range(self.dim[0]) for j in range(self.dim[1]))]])

    def copy(self):
        lst = [[c for c in r] for r in self.data]
        return Matrix(lst)

    def Kronecker_product(self, other):
        ans_mat = Matrix(dim=(self.dim[0] * other.dim[0], self.dim[1] * other.dim[1]))
        for r in range(self.dim[0]):
            for c in range(self.dim[1]):
                num_lst = [[0] * other.dim[0] for _ in range(other.dim[0])]
                for i in range(other.dim[0]):
                    num_lst[i][i] = self.data[r][c]
                num_mat = Matrix(num_lst)
                ans_mat[r * other.dim[0] : (r+1) * other.dim[0], c * other.dim[1] : (c+1) * other.dim[1]] = num_mat.dot(other)
        return ans_mat
    
    def __getitem__(self, key):
        if type(key[0]) == int and type(key[1]) == int:
            # 行和列都只有1个元素
            return self.data[key[0]][key[1]]
        elif type(key[0]) == int and type(key[1]) == slice:
            # 行为一个元素，列为列表切片
            ans_mat = Matrix([self.data[key[0]][key[1]]])
            return ans_mat
        elif type(key[0]) == slice and type(key[1]) == int:
            # 行数为列表切片，列为一个元素。
            ans_lst = []
            ans_mat = None
            lst = list(range(self.dim[0]))
            lst_slice = lst[key[0]]
            for r in lst_slice:
                ans_lst.extend([[self.data[r][key[1]]]])
            ans_mat = Matrix(ans_lst)
            return ans_mat
        elif type(key[0]) == slice and type(key[1]) == slice:
            # 行列都是切片
            ans_lst = []
            ans_mat = None
            lst = list(range(self.dim[0]))
            lst_slice = lst[key[0]]
            for r in lst_slice:
                ans_lst.extend([self.data[r][key[1]]])
            ans_mat = Matrix(ans_lst)
            return ans_mat
        else:
            print("Error")
            return 0

    def __setitem__(self, key, value):
        if type(key[0]) == int and type(key[1]) == int:
            # 行和列都只有1个元素
            self.data[key[0]][key[1]] = value
            return 
        elif type(key[0]) == int and type(key[1]) == slice:
            # 行为一个元素，列为列表切片
            self.data[key[0]][key[1]] = value.data[0][:]
            return 
        elif type(key[0]) == slice and type(key[1]) == int:
            # 行数为列表切片，列为一个元素。
            lst = list(range(self.dim[0]))
            lst_slice = lst[key[0]]
            for r in lst_slice:
                self.data[r][key[1]] = value.data[r][0]
            return 
        elif type(key[0]) == slice and type(key[1]) == slice:
            # 行列都是切片
            lst = list(range(self.dim[0]))
            lst_slice = lst[key[0]]
            x = 0
            for r in lst_slice:
                self.data[r][key[1]] = value.data[x]
                x += 1
            return
    
    def __pow__(self, n):
        if self.dim[0] != self.dim[1]:
            print("Not a square matrix!")
            return Matrix([[0]])
        else:
            ans = I(self.dim[0])
            for _ in range(n):
                ans = ans.dot(self)
        return ans
    
    def __add__(self, other):
        if self.dim != other.dim:
            print("矩阵维数不同，无法相加！")
            return Matrix(dim=(1,1))
        else:
            ans_lst = [[0] * self.dim[1] for _ in range(self.dim[0])]
            for r in range(self.dim[0]):
                for c in range(self.dim[1]):
                    ans_lst[r][c] = self.data[r][c] + other.data[r][c]
            ans_mat = Matrix(ans_lst)
            return ans_mat

    def __sub__(self, other):
        if self.dim != other.dim:
            print("矩阵维数不同，无法相减！")
            return Matrix(dim=(1,1))
        else:
            ans_lst = [[0] * self.dim[1] for _ in range(self.dim[0])]
            for r in range(self.dim[0]):
                for c in range(self.dim[1]):
                    ans_lst[r][c] = self.data[r][c] - other.data[r][c]
            ans_mat = Matrix(ans_lst)
            return ans_mat

    def __mul__(self, other):
        if self.dim != other.dim:
            print("矩阵维数不同，无法相乘！")
            return Matrix(dim=(1,1))
        else:
            ans_lst = [[0] * self.dim[1] for _ in range(self.dim[0])]
            for r in range(self.dim[0]):
                for c in range(self.dim[1]):
                    ans_lst[r][c] = self.data[r][c] * other.data[r][c]
            ans_mat = Matrix(ans_lst)
            return ans_mat
    
    def __len__(self):
        return self.dim[0] * self.dim[1]
    
    def __str__(self):  # 未解决"-0.0"的问题
        wid = 1
        add_wid = 2
        for r in range(self.dim[0]):
            for c in range(self.dim[1]):
                if self.data[r][c] == int(self.data[r][c]):
                    self.data[r][c] = int(self.data[r][c])
                w_x = len(str(self.data[r][c]))
                if isinstance(self.data[r][c], int) and w_x > wid:
                    wid = w_x
                elif w_x > 5 and wid <= 5:
                    add_wid = 4
                    wid = 5
                elif w_x > wid:
                    add_wid = 4
                    wid = w_x
        display = "["
        for r in range(self.dim[0]):
            if r > 0:
                display += ' '
            display += '[' + ' '.join("{:>{width}}".format(round(x, 5), width=wid + add_wid) for x in self.data[r]) + ']'
            if r < self.dim[0] - 1:
                display += "\n"
        display += "]"
        return display

    def det(self):
        if self.dim[0] != self.dim[1]:
            print("三行四列的行列式我没有见过啊！")
            return 0
        else:          
            ans = 1            
            ans_mat = self.copy()
            for i in range(self.dim[0] - 1):
                if not ans_mat[i, i]:
                    flag = False
                    for k in range(i+1,self.dim[0]):
                        if ans_mat[k,i]:
                            #交换两行
                            memory = ans_mat[i,:]
                            ans_mat[i,:] = ans_mat[k,:]
                            ans_mat[k,:] = memory
                            flag = True
                            ans *= -1
                            break
                    if not flag:
                        return 0
                for j in range(i + 1, self.dim[0]):    
                    factor = ans_mat[j, i] / ans_mat[i, i]
                    f_mat = Matrix(dim = ans_mat[i, i:].dim, init_value = factor)
                    ans_mat[j, i:] -= f_mat * ans_mat[i, i:]

            for i in range(self.dim[0]):
                ans *= ans_mat.data[i][i]
            if int(ans) == ans:
                return int(ans)
            return ans

    def inverse(self):
        dime = self.dim[0]
        joint_mat = Matrix(dim=(dime , dime* 2))
        joint_mat[:, :dime] = self
        joint_mat[:, dime: ] = I(dime)
        # 高斯消元法
        for i in range(self.dim[0] - 1):
            if not joint_mat[i, i]:
                flag = False
                for k in range(i+1,self.dim[0]):
                    if joint_mat[k,i]:
                        #交换两行
                        memory = joint_mat[i,:]
                        joint_mat[i,:] = joint_mat[k,:]
                        joint_mat[k,:] = memory
                        flag = True
                        break
                if not flag:
                    print("该矩阵为奇异阵，没有逆矩阵")
                    return Matrix([[0]])
            for j in range(i + 1, dime):
                factor = joint_mat[j, i] / joint_mat[i, i]
                f_mat = Matrix(dim = joint_mat[i, i:].dim, init_value=factor)
                joint_mat[j, i:] -= f_mat * joint_mat[i, i:]
        for i in range(dime - 1, -1, -1):
            for j in range(i):
                factor = joint_mat[j, i] / joint_mat[i, i]
                f_mat = Matrix(dim = joint_mat[i, i:].dim, init_value=factor)
                joint_mat[j, i:] -= f_mat * joint_mat[i, i:]
            unify_factor = 1 / joint_mat[i, i]
            u_mat = Matrix(dim=joint_mat[i, :].dim, init_value=unify_factor)
            joint_mat[i, :] *= u_mat
        inv_mat = joint_mat[:,dime :]
        return inv_mat
        
    def rank(self):
        ans_mat = self.copy()
        for i in range(self.dim[0] - 1):
            for j in range(i + 1, self.dim[0]):
                if ans_mat[i, i]:
                    factor = ans_mat[j, i] / ans_mat[i, i]
                    f_mat = Matrix(dim = ans_mat[i, i:].dim, init_value = factor)
                    ans_mat[j, i:] -= f_mat * ans_mat[i, i:]
        r = 0
        for i in range(self.dim[0]):
            if ans_mat.data[i][self.dim[1] - 1]:
                r += 1
        return r


def I(n):
    ans_lst = [[0] * n for _ in range(n)]
    for i in range(n):
        ans_lst[i][i] = 1
    return Matrix(ans_lst)

def narray(dim, init_value=1): # dim (,,,,,), init为矩阵元素初始值
    return Matrix(dim=dim, init_value=init_value)

def arange(start, end, step=1):
    mat_lst = list(range(start, end, step))
    return Matrix([mat_lst])
    
def zeros(dim):
    return Matrix(dim=dim)

def zeros_like(matrix):
    return Matrix(dim=matrix.dim)

def ones(dim):
    return Matrix(dim=dim, init_value=1)

def ones_like(matrix):
    return Matrix(dim=matrix.dim, init_value=1)

def nrandom(dim):
    lst = [[random.uniform(0, 1) for c in range(dim[1])] for r in range(dim[0])]
    return Matrix(lst)

def nrandom_like(matrix):
    lst = [[random.uniform(0, 1) for c in range(matrix.dim[1])] for r in range(matrix.dim[0])]
    return Matrix(lst)

def concatenate(items, axis=0):
    try:
        if not items:
            raise ValueError("缺少拼接对象")
        for m in items:
            if type(m) != Matrix:
                raise ValueError("拼接对象不是矩阵")
        if axis != 0 and axis != 1:
            raise ValueError("无法在此维度上拼接")
        if axis == 0:
            col = items[0].dim[1]
            for m in items:
                if m.dim[1] != col:
                    raise ValueError("矩阵形状不匹配")
            # 行数增加
            row_num = items[0].dim[0]
            ans_mat = zeros((len(items) * row_num, items[0].dim[1]))
            for i in range(len(items)):
                ans_mat[i * row_num : (i + 1) * row_num, :] = items[i]
            return ans_mat
        
        if axis == 1:
            row = items[0].dim[0]
            for c in items:
                if c.dim[0] != row:
                    raise ValueError("矩阵形状不匹配")
            # 列数增加
            col_num = items[0].dim[1]
            ans_mat = zeros((items[0].dim[0], len(items) * col_num))
            for i in range(len(items)):
                ans_mat[ : , i * col_num : (i + 1) * col_num] = items[i]
            return ans_mat
            
    except Exception as e:
        print(e)
        return Matrix([[0]])

def vectorize(func):
    def v_func(mat):
        if isinstance(mat, Matrix):
            ans_mat = mat.copy()
            for r in range(ans_mat.dim[0]):
                for c in range(ans_mat.dim[1]):
                    ans_mat.data[r][c] = func(mat.data[r][c])
            return ans_mat
        else:
            return func(mat)
    return v_func

if __name__ == "__main__":
    print("test here")
    
    # Create a random 5x5 matrix
    m1 = Matrix([[random.randint(0, 100) for i in range(5)] for _ in range(5)])
    print("Matrix m1:")
    print(m1)
    
    # Test matrix multiplication with its inverse
    print("\nMatrix m1 * m1.inverse():")
    print(m1.dot(m1.inverse()))
    
    # Test matrix reshaping
    m2 = m1.reshape((1, 25))
    print("\nReshaped matrix m2:")
    print(m2)
    
    # Test matrix transpose
    m3 = m1.T()
    print("\nTransposed matrix m3:")
    print(m3)
    
    # Test matrix sum along axis
    print("\nMatrix m1 sum along axis 0:")
    print(m1.sum(axis=0))
    print("\nMatrix m1 sum along axis 1:")
    print(m1.sum(axis=1))
    print("\nTotal sum of matrix m1:")
    print(m1.sum())
    
    # Test matrix Kronecker product
    m4 = Matrix([[1, 2], [3, 4]])
    print("\nMatrix m1 Kronecker product with m4:")
    print(m1.Kronecker_product(m4))
    
    # Test matrix power
    m5 = m1**2
    print("\nMatrix m1 squared:")
    print(m5)
    
    # Test matrix addition and subtraction
    m6 = m1 + m1
    print("\nMatrix m1 + m1:")
    print(m6)
    
    m7 = m1 - m1
    print("\nMatrix m1 - m1:")
    print(m7)
    
    # Test matrix multiplication element-wise
    m8 = m1 * m1
    print("\nMatrix m1 * m1 element-wise:")
    print(m8)
    
    # Test matrix determinant
    print("\nDeterminant of m1:")
    print(m1.det())
    
    # Test matrix inverse
    m9 = m1.inverse()
    print("\nInverse of m1:")
    print(m9)
    
    # Test matrix rank
    print("\nRank of m1:")
    print(m1.rank())
    
    # Test identity matrix creation
    identity_matrix = I(3)
    print("\nIdentity matrix:")
    print(identity_matrix)
    
    # Test creating a matrix with specified dimensions and initial value
    m10 = narray((2, 3), init_value=5)
    print("\nMatrix m10 with dimensions (2, 3) and initial value 5:")
    print(m10)
    
    # Test creating a matrix with random values
    m11 = nrandom((2, 2))
    print("\nRandom matrix m11:")
    print(m11)
    
    # Test vectorized function
    @vectorize
    def square(x):
        return x**2
    
    m12 = square(m1)
    print("\nMatrix m1 after squaring each element:")
    print(m12)
    
    # Test concatenation along axis
    m13 = concatenate((m1, m2), axis=0)
    print("\nConcatenated matrix along axis 0:")
    print(m13)
    
    m14 = concatenate([m1, m3], axis=1)
    print("\nConcatenated matrix along axis 1:")
    print(m14)

