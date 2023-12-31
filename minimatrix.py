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
            return 

    def dot(self, other):
        # 注意到在不符合运算规则时，输出为None，在复合运算中可能导致程序报错
        # 因此在需要的函数开头，先检查所有矩阵参数是否为空
        # 下同
        if not isinstance(other,Matrix):
            return
    
        if self.dim[1] != other.dim[0]:
            print("维数错误，无法相乘。")
            return 
        lst_mat = [[sum(self[r, i] * other [i, c] for i in range(self.dim[1])) 
                    for c in range(other.dim[1])] for r in range(self.dim[0])]
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
        if not isinstance(other,Matrix):
            return
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
            print("错误。")
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
            print("矩阵不是方阵。")
            return
        else:
            ans = I(self.dim[0])
            n_ = n
            b = self
            while n_:
                if n_ & 1:
                    ans = ans.dot(b)
                b = b.dot(b)
                n_ >>= 1
            
        return ans
    
    def __add__(self, other):
        if not isinstance(other,Matrix):
            return
        if self.dim != other.dim:
            print("矩阵维数不同，无法相加！")
            return
        else:
            ans_lst = [[self.data[r][c] + other.data[r][c] for c in range(self.dim[1])]
                       for r in range(self.dim[0])]
            return Matrix(ans_lst)

    def __sub__(self, other):
        if not isinstance(other,Matrix):
            return
        if self.dim != other.dim:
            print("矩阵维数不同，无法相减！")
            return
        else:
            ans_lst = [[self.data[r][c] - other.data[r][c] for c in range(self.dim[1])]
                        for r in range(self.dim[0])]
            return Matrix(ans_lst)

    def __mul__(self, other):
        if not isinstance(other,Matrix):
            return
        if self.dim != other.dim:
            print("矩阵维数不同，无法相乘！")
            return
        else:
            ans_lst = [[self.data[r][c] * other.data[r][c] for c in range(self.dim[1])]
                        for r in range(self.dim[0])]
            return Matrix(ans_lst)
    
    def __len__(self):
        return self.dim[0] * self.dim[1]
    
    def __str__(self):
        wid = 0
        for r in range(self.dim[0]):
            for c in range(self.dim[1]):
                if self.data[r][c] == int(self.data[r][c]):
                    self.data[r][c] = int(self.data[r][c])
                # 如果存在大整数，宽度以最大整数为准
                w_x = len(str(self.data[r][c]))
                if isinstance(self.data[r][c], int):
                    wid = max(w_x, wid)
                # 若最大整数宽度不超过5，且存在浮点数，则宽度以7为准
                elif w_x > 7:
                    wid = 7
                    break
                else:
                    wid = min(wid, 7, w_x)
        wid += 1
        display = "["
        for r in range(self.dim[0]):
            if r > 0:
                display += ' '
            display += '[' + ' '.join("{:>{width}}".format(round(x, 5), width=wid) for x in self.data[r]) + ']'
            if r < self.dim[0] - 1:
                display += "\n"
        display += "]"
        return display

    def det(self):
        if self.dim[0] != self.dim[1]:
            print("三行四列的行列式我没有见过啊！")
            return
        else:          
            ans = 1            
            ans_mat = self.copy()
            # 调用gauss消元方法
            ans_mat, seq = ans_mat.Gauss_elimination()
            for i in range(self.dim[0]):
                ans *= ans_mat.data[i][i]
            ans *= (-1)**seq
            if int(ans) == ans:
                return int(ans)
            return ans * (-1)**seq

    def inverse(self):
        if self.dim[0] != self.dim[1]:
            print("该矩阵不是方阵，没有逆矩阵。")
            return
        dime = self.dim[0]
        joint_mat = Matrix(dim=(dime , dime* 2))
        joint_mat[:, :dime] = self
        joint_mat[:, dime: ] = I(dime)
        # 高斯消元法
        joint_mat = joint_mat.Gauss_elimination()[0]
        if not any(x for x in joint_mat.data[dime - 1][:dime]):
            print("该矩阵为奇异阵，没有逆矩阵。")
            return
        for i in range(dime - 1, -1, -1):
            unify_factor = 1 / joint_mat[i, i]
            u_mat = Matrix(dim=joint_mat[i, :].dim, init_value=unify_factor)
            joint_mat[i, :] *= u_mat
        inv_mat = joint_mat[:,dime :]
        return inv_mat
        
    def Gauss_elimination(self):
        seq = 0
        op_mat = self.copy()
        for i in range(min(self.dim)): # 取行数和列数的最小值
            flag = True
            pivot = i
            while flag and not op_mat.data[i][pivot]:  # 寻找当前pivot列的非零元
                for i1 in range(i+1, self.dim[0]):
                    if op_mat.data[i1][pivot]:
                        op_mat[i, :], op_mat[i1, :] = op_mat[i1, :], op_mat[i, :] #交换两行
                        flag = False
                        seq += 1
                        break
                if flag: # 当前pivot列再i行以下只有零元，考察下一个pivot列
                    pivot += 1
                if pivot >= op_mat.dim[1]:
                    return op_mat, seq
            else:
                for j in range(self.dim[0]):
                    if j != i:
                        factor = op_mat[j, pivot] / op_mat[i, pivot]
                        f_mat = Matrix(dim = op_mat[i, :].dim, init_value = factor)
                        op_mat[j, :] -= f_mat * op_mat[i, :]
            
        return op_mat, seq
        
        
    def rank(self):
        ans_mat = self.Gauss_elimination()[0]
        r = 0
        for i in range(self.dim[0]):
            if any(x for x in ans_mat.data[i]):
                r += 1
        return r

    def equation(self,other):
        if not isinstance(other,Matrix):
            return

        #检验数据是否正确
        if other.dim[1] != 1 or other.dim[0] != self.dim[0]:
            print("数据错误。")
            return
        #生成增广矩阵
        b = Matrix([self.data[i]+other.data[i] for i in range(self.dim[0])])
        if self.rank() < b.rank():
            print("无解。")
            return
        if self.rank() < self.dim[1]:
            print("有无穷多组解。")
            return
        else:
            return self.inverse().dot(other)

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
    if not items:
        print("缺少拼接对象")
        return None

    for m in items:
        if type(m) != Matrix:
            print("拼接对象不是矩阵")
            return None

    if axis != 0 and axis != 1:
        print("无法在此维度上拼接")
        return None

    if axis == 0:
        col = items[0].dim[1]
        for m in items:
            if m.dim[1] != col:
                print("矩阵形状不匹配")
                return None
        # 行数增加
        row_num = items[0].dim[0]
        ans_mat = zeros((len(items) * row_num, items[0].dim[1]))
        for i in range(len(items)):
            ans_mat[i * row_num: (i + 1) * row_num, :] = items[i]
        return ans_mat

    if axis == 1:
        row = items[0].dim[0]
        for c in items:
            if c.dim[0] != row:
                print("矩阵形状不匹配")
                return None
        # 列数增加
        col_num = items[0].dim[1]
        ans_mat = zeros((items[0].dim[0], len(items) * col_num))
        for i in range(len(items)):
            ans_mat[:, i * col_num: (i + 1) * col_num] = items[i]
        return ans_mat

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
    # All test code are presented in main.py