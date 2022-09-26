import numpy as np

## 1. Zeros and Ones

# a, b, c = map(int, input().split())
# print(np.zeros((a,b,c), dtype = np.int))
# print(np.ones((a,b,c), dtype = np.int))  

'''input
3 3 3
'''
'''output
[[[0 0 0]
  [0 0 0]
  [0 0 0]]

 [[0 0 0]
  [0 0 0]
  [0 0 0]]

 [[0 0 0]
  [0 0 0]
  [0 0 0]]]
[[[1 1 1]
  [1 1 1]
  [1 1 1]]

 [[1 1 1]
  [1 1 1]
  [1 1 1]]

 [[1 1 1]
  [1 1 1]
  [1 1 1]]]
'''

#-----------------------------
## 2. Arrays
# a = list(map(int, input().split()))
# b = np.array(a, float)
# b = np.flip(b)
# print(b)

'''input
1 2 3 4 -8 -10
'''
'''output
[-10.  -8.   4.   3.   2.   1.]
'''

#-----------------------------
## 3. Shape and reshape

# a = input()
# a = a.split()
# print(np.array(a, float).reshape(3,3))

'''input
1 2 3 4 5 6 7 8 9
'''
'''output
[[1 2 3]
 [4 5 6]
 [7 8 9]]
'''

#-----------------------------
## 4. Array Mathematics

# a = input()
# a = a.split()

# for i in range(int(a[0])):
#     k = input()
#     k = k.split()
#     b = input()
#     b = b.split()
    
# k = [1, 2, 3, 4]
# b = [5, 6, 7, 8] 
# k = np.array(k, int)
# b = np.array(b, int)
# k = k.reshape(1,int(a[1]))
# b = b.reshape(1,int(a[1]))

# print(np.add(k,b))
# print(np.subtract(k,b))
# print(np.multiply(k,b))
## print(np.divide(k,b))
# ad = np.divide(k, b)
# ad = list(map(np.round, ad))
# print(ad)
# print(np.mod(k,b))
# print(np.power(k,b))    

'''input
1 4
1 2 3 4
5 6 7 8
'''
'''output
[ 6  8 10 12]
[-4 -4 -4 -4]
[ 5 12 21 32]
[0 0 0 0]
[1 2 3 4]
[    1    64  2187 65536]
'''

#-----------------------------
## 4. Sum and Prod

aa = input().split()
arr = []
for n in range(int(aa[0])):
    lst = input()
    lst = lst.split()
    arr.append(lst)
# print(arr)  # [['1', '2'], ['3', '4']]

arr = np.array(arr, int)    # size=(int(aa[0]), int(aa[1]))
print(np.prod(np.sum(arr, axis = 0)))

'''input
2 2
1 2
3 4
'''
'''output
24
'''
