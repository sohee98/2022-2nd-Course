## 1. Find a String

# def count_substring(string, sub_string):
#     # put your code here...
#     count = 0
#     length = len(string) - len(sub_string)
#     for i in range(length):
#         if string[i:i + len(sub_string)] == sub_string:
#             count += 1
#     return count

# if __name__ == '__main__':
#     string = input().strip()
#     sub_string = input().strip()

#     count = count_substring(string, sub_string)
#     print(count)

''' input
ABCDCDC
CDC
'''
'''output
2
'''
#-----------------------------
## 2. Compress the String
import pdb

# a = ()
# string_input = input()
# # a = itertools.groupby(string_input)
# lst = []

# for letter in string_input:
#     lst.append(letter)
# aa = sorted(set(list(lst)))
# # pdb.set_trace()
# for i in aa:
#     bb = lst.count(i)
#     a = (int(i), bb)
#     print(a, end=' ')

''' input
1222311
'''
'''output
(1, 1) (2, 3) (3, 1) (1, 2)
'''
#-----------------------------
## 3. Word Order
# put your code here...
# import pdb
# cnt = int(input())
# a = []
# for i in range(cnt):
#   str = input()
#   a.append(str)
#   # pdb.set_trace()
# kind = sorted(set(a))
# print(len(kind))
## for string in kind:
##   print(a.count(string),end=' ')

# group = []
# for i in range(len(a)):
#   cnt = 0
#   if a[i] not in group:
#     for c in range(len(a)-i):
#       if a[i] == a[i+c]:
#         cnt +=1
#     group.append(a[i])
#     print(cnt, end=' ')

### Dict
# cnt = int(input())
# words = {}
# for c in range(cnt):
#   ip = input()
#   if ip in words:
#     words[ip] += 1
#   else:
#     words[ip] = 1
# # print(words)
# for k, v in words.items():
#   print(v, end=" ")

''' input
4
bcdef
abcdefg
bcde
bcdef
'''
'''output
3
2 1 1
'''
#-----------------------------
## 4. No Idea!
# s = list(map(int, input().split()))
# L = list(map(int, input().split()))
# A = list(map(int, input().split()))
# B = list(map(int, input().split()))

# result = 0
# for  i in L:
#   if i in A:
#     result += 1
#   if i in B:
#     result -= 1
# print(result)

''' input
3 2
1 5 3
3 1
5 7
'''
'''output
1
'''