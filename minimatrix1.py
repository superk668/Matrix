import random

class Matrix:
    def __init__(self, data=None, dim=None, init_value=0):
        if data:
            self.data = data
            row = len(data)
            column = len(data[0])
            self.dim = (row, column)
        elif dim:
            self.dim = dim
            self.data = [[init_value for c in range(dim[1])] for r in range(dim[0])]
        else:
            print("你没有提供任何数据！")
            del self
    
    
    def __str__(self):
        display = "["
        for r in range(self.dim[0]):
            if r > 0:
                display += ' '
            display += '[' + ' '.join(f"{x:4}" for x in self.data[r]) + ']'
            if r < self.dim[0] - 1:
                display += "\n"
        display += "]"
        return display
    
    def shape(self):
        return self.dim
    
    "unfinished"
    def reshape(self, newdim):
        pass
    
    def dot(self, other):
        if self.dim[1] != other.dim[0]:
            print("These 2 matrices has no point product.")
            return
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
        return Matrix(self.data)

    def Kronecker_product(self, other):
        ans_mat = Matrix(dim=(self.dim[0] * other.dim[0], self.dim[1] * other.dim[1]))
        for r in range(self.dim[0]):
            for c in range(self.dim[1]):
                ans_mat[r * other.dim[0] : (r+1) * other.dim[0], c * other.dim[1] : (c+1) * other.dim[1]] = Matrix(dim=other.dim, init_value=self.data[r][c]).dot(other)
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
            return "Error"

    def __setitem__(self, key, value):
        if type(key[0]) == int and type(key[1]) == int:
            # 行和列都只有1个元素
            self.data[key[0]][key[1]] = value
            return 
        elif type(key[0]) == int and type(key[1]) == slice:
            # 行为一个元素，列为列表切片
            self.data[key[0]][key[1]] = value.data[0][key[1]]
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
            c = 0
            for r in lst_slice:
                self.data[r][key[1]] = value.data[c]
                c += 1
            return 



mat1 = Matrix([[1,2], [4,5], [0,9]])
mat2 = Matrix([[-2, 1, 3], [-4, 0, 9], [2, 1, 5]])
print(mat1.Kronecker_product(mat2))

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

