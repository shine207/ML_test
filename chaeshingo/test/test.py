import numpy as np

gibo_size = 3
info_data_size = 2
gibo_data = np.zeros((gibo_size, gibo_size, info_data_size))
gibo_data[0][1] = [2,3]
print(gibo_data)
#print( gibo_data.flatten() )

def aaa(a):
    a = 1

aaa(gibo_data)
print(gibo_data)
