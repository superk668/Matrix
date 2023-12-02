import random
import decimal

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
    
    "unfinished"
    def reshape(self, newdim):
        pass
    
    def dot(self, other):
        if self.dim[1] != other.dim[0]:
            print("These 2 matrices has no point product.")
            return Matrix([[0]])
        lst_mat = [[0 for c in range(other.dim[1])] for r in range(self.dim[0])]
        # ans_mat = Matrix(dim=(self.dim[0], other.dim[1]))
        # ans_mat[i][j]
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
            ans_lst = [0 for _ in range(self.dim[1])]
            for c in range(self.dim[1]):
                for r in range(self.dim[0]):
                    ans_lst[c] += self.data[r][c]
            ans_mat = Matrix([ans_lst])
            return ans_mat
        elif axis == 1:
            ans_lst = []
            for r in range(self.dim[0]):
                ans_lst.append([sum(self.data[r])])
            ans_mat = Matrix(ans_lst)
            return ans_mat
        elif axis == None:
            tot = 0
            for r in range(self.dim[0]):
                tot += sum(self.data[r])
            return Matrix([[tot]])

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
        display = "["
        for r in range(self.dim[0]):
            if r > 0:
                display += ' '
            display += '[' + ' '.join(f"{round(x, 6):4}" for x in self.data[r]) + ']'
            if r < self.dim[0] - 1:
                display += "\n"
        display += "]"
        return display

    def det(self):
        if self.dim[0] != self.dim[1]:
            print("三行四列的行列式我没有见过啊！")
            return 0
        else:
            ans_mat = self.copy()
            for i in range(self.dim[0] - 1):
                for j in range(i + 1, self.dim[0]):
                    factor = ans_mat[j, i] / ans_mat[i, i]
                    f_mat = Matrix(dim = ans_mat[i, i:].dim, init_value = factor)
                    ans_mat[j, i:] -= f_mat * ans_mat[i, i:]
            ans = 1
            for i in range(self.dim[0]):
                ans *= ans_mat.data[i][i]
            return ans

    def inverse(self):
        if not self.det():
            print("该矩阵为奇异阵，没有逆矩阵")
            return Matrix(dim=(1,1))
        else:
            dime = self.dim[0]
            joint_mat = Matrix(dim=(dime , dime* 2))
            joint_mat[:, :dime] = self
            joint_mat[:, dime :] = I(dime)
            # 高斯消元法
            for i in range(dime - 1):
                for j in range(i + 1, dime):
                    factor = joint_mat[j, i] / joint_mat[i, i]
                    f_mat = Matrix(dim = joint_mat[i, i:].dim, init_value=factor)
                    joint_mat[j, i:] -= f_mat * joint_mat[i, i:]
            for i in range(dime - 1, 0, -1):
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
    return Matrix(dim=dim, init_value=1)

def arange(start, end, step):
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





mat1 = Matrix([[1,2,3],[1,3,5],[0,-3,1]])

'''print(mat1)
print(mat1.rank())
print(mat1.inverse())
print(mat1.dot(mat1.inverse()))'''

'''print(I(4))
mat1 = Matrix([[1,2], [3, 4]])
print(mat1 ** 0)'''

'''mat1 = Matrix([[1,2], [3, 4]])
mat2 = Matrix([[5,6,7],[8,9,10]])
print(mat1.Kronecker_product(mat2))
print(mat2.Kronecker_product(mat1))'''

'''mat1 = Matrix([[1,2], [4,5], [0,9]])
print(mat1.sum(1))'''

'''mat1 = Matrix([[1,2], [4,5], [0,9]])
print(mat1)
print(mat1.T())'''

'''mat1 = Matrix([[1,2,3],[1,3,5],[0,-3,1]])
mat1[:, :] = Matrix(dim=(3, 3))
print(mat1)'''

"""mat1 = Matrix([[1,2,3],[1,3,5],[0,-3,1]])
mat2 = Matrix([[-2, 1, 3], [-4, 0, 9], [2, 1, 5]])
print(mat1.dot(mat2))
"""

