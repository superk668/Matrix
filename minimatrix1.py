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
            quit()
        
        
    def __str__(self):
        return str(self.data)
    
    def shape(self):
        return self.dim
    
    def reshape(self, newdim):
        pass
    




mat2 = Matrix()

print(mat2.shape())