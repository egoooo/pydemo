import numpy as np

# numpyDEmo.empty 方法用来创建一个指定形状（shape）、数据类型（dtype）且未初始化的数组：
# numpyDEmo.empty(shape, dtype = float, order = 'C')
# 数组元素为随机值，因为它们未初始化。
x = np.empty([3,2], dtype=int)
print(x)
# [[       18         0]
#  [        0         0]
#  [        0 409217636]]


# numpy.zeros
# 创建指定大小的数组，数组元素以 0 来填充：
# umpy.zeros(shape, dtype = float, order = 'C')

# 默认为浮点数
x = np.zeros(5)
print(x)
#[0. 0. 0. 0. 0.]

# 设置类型为整数
y = np.zeros(5, dtype=np.int)
print(y)
# [0 0 0 0 0]

# 自定义类型
z = np.zeros((2, 2), dtype=[('x', 'i4'), ('y', 'i4')])
print(z)
# [[(0, 0) (0, 0)]
#  [(0, 0) (0, 0)]]

# numpy.ones
# # 创建指定形状的数组，数组元素以 1 来填充：
# # numpy.ones(shape, dtype = None, order = 'C')

# 自定义类型
x = np.ones([2,2], dtype = int)
print(x)
# [[1 1]
#  [1 1]]


a = np.arange(24).reshape((2, 3, 4))
print(a)
# [[[ 0  1  2  3]
#   [ 4  5  6  7]
#   [ 8  9 10 11]]
#
#  [[12 13 14 15]
#   [16 17 18 19]
#   [20 21 22 23]]]

# ndarray类
#
#   NumPy中的数组类被称为ndarray，要注意的是numpy.array与Python标准库中的array.array是不同的。ndarray具有如下比较重要的属性：
#
# ndarray.ndim
#
#   ndarray.ndim表示数组的维度。
#
# ndarray.shape
#
#   ndarray.shape是一个整型tuple，用来表示数组中的每个维度的大小。例如，对于一个n行和m列的矩阵，其shape为(n,m)。
#
# ndarray.size
#
#   ndarray.size表示数组中元素的个数，其值等于shape中所有整数的乘积。
#
# ndarray.dtype
#
#   ndarray.dtype用来描述数组中元素的类型，ndarray中的所有元素都必须是同一种类型，如果在构造数组时，传入的参数不是同一类型的，不同的类型将进行统一转化。除了标准的Python类型外，NumPy额外提供了一些自有的类型，如numpy.int32、numpy.int16以及numpy.float64等。
#
# ndarray.itemsize
#
#   ndarray.itemsize用于表示数组中每个元素的字节大小。

a = np.arange(15).reshape(3,5)
print(a)
# [[ 0  1  2  3  4]
#  [ 5  6  7  8  9]
#  [10 11 12 13 14]]

print(a.shape)
# (3, 5)
print(a.ndim)
# 2

print(a.dtype.name)
# int32
print(a.dtype)
# int32
print(a.size)
# 15
print(a.itemsize)
# 4

print(type(a))
# <class 'numpy.ndarray'>


#  NumPy中创建数组,利用Python中常规的list和tuple进行创建。
a = np.array([1,2,3,4,5,6])
b = np.array((1,2,3,4,5,6))
print(a) #[1 2 3 4 5 6]
print(b) #[1 2 3 4 5 6]

#传入的参数必须是同一结构,不是同一结构将发生转换。
a = np.array([1,2,3.5])
c = np.array(['1',2,3])
print(a) #[1.  2.  3.5]
print(c) #['1' '2' '3']

b = np.array([[1,2,3],[2,3,4],[3,4,5]])
print(b)
# [[1 2 3]
#  [2 3 4]
#  [3 4 5]]


# 基本操作
# 对数组中的算术操作是元素对应（elementwise）的
# 例如，对两个数组进行加减乘除
# 其结果是对两个数组对一个位置上的数进行加减乘除
# 数组算术操作的结果会存放在一个新建的数组中
a = np.array([10,20,30,40])
b = np.arange(1,5)
# numpy 包中的使用 arange 函数创建数值范围并返回 ndarray 对象，函数格式如下：
# numpy.arange(start, stop, step, dtype)
# 根据 start 与 stop 指定的范围以及 step 设定的步长，生成一个 ndarray。

print(b)
# [1 2 3 4]
print(a+b)
# [11 22 33 44]
print(a-b)
# [ 9 18 27 36]
print(a*b)
# [ 10  40  90 160]
print(a/b)
# [10. 10. 10. 10.]
print(a*3)
# [ 30  60  90 120]
print(a<21)
# [ True  True False False]


# 在NumPy中，
# *用于数组间元素对应的乘法，
# 而不是矩阵乘法，矩阵乘法可以用dot()方法来实现。

A = np.array([[1,2],[3,4]])
B = np.array([[0,1],[0,1]])

print(A*B)
# [[0 2]
#  [0 4]]
print(A.dot(B))
# [[0 3]
#  [0 7]]

#
# 有些操作
# ，如*=，+=，-=，/=等操作，
# 会直接改变需要操作的数组，
# 而不是创建一个新的数组。

a = np.ones((2,3), dtype = int)
b = np.random.random((2,3))
print(a)
# [[1 1 1]
#  [1 1 1]]
print(b)
# [[0.23319244 0.76184239 0.73300245]
#  [0.63151206 0.48584914 0.85245899]]
a*=3
print(a)
# [[3 3 3]
#  [3 3 3]]

#当操作不同类型的数组时，最终的结果数组的类型取决于精度最宽的数组的类型。

a=np.arange(10)
print(a)
# [0 1 2 3 4 5 6 7 8 9]
print(a.sum())
# 45
print(a.max())
# 9
print(a.min())
# 0
print(np.exp(a))
# [1.00000000e+00 2.71828183e+00 7.38905610e+00 2.00855369e+01
#  5.45981500e+01 1.48413159e+02 4.03428793e+02 1.09663316e+03
#  2.98095799e+03 8.10308393e+03]
print(np.sqrt(a))
# [0.         1.         1.41421356 1.73205081 2.         2.23606798
#  2.44948974 2.64575131 2.82842712 3.        ]
print(np.sin(a))
# [ 0.          0.84147098  0.90929743  0.14112001 -0.7568025  -0.95892427
#  -0.2794155   0.6569866   0.98935825  0.41211849]

#数组索引和迭代
#  冒号 : 的解释：如果只放置一个参数，
# 如 [2]，将返回与该索引相对应的单个元素。
# 如果为 [2:]，表示从该索引开始以后的所有项都将被提取.
# 如果使用了两个参数，如 [2:7]，
# 那么则提取两个索引(不包括停止索引)之间的项。
print(a[3]) #3
print(a[-1]) #9 负的index表示，从后往前。-1表示最后一个元素
print(a[2:5]) #[2 3 4]
print(a[2:7:2]) #[2 4 6]
print(a[::2]) #[0 2 4 6 8]


a = np.array([[1,2,3],[3,4,5],[4,5,6]])
print(a)
# 从某个索引处开始切割
print('从数组索引 a[1:] 处开始切割')
print(a[1:])
# 从数组索引 a[1:] 处开始切割
# [[3 4 5]
#  [4 5 6]]

#相对于一维数组而言，二维（多维）数组用的会更多。
# 一般语法是arr_name[行操作, 列操作]
arr = np.arange(12).reshape((3, 4))
print(arr)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
print(arr[:,0])
# [0 4 8] 取第0列的数据，以行的形式返回的
print(arr[:,0:1])
# [[0]
#  [4]
#  [8]] 取第0列的数据，以列的形式返回的
print(arr[1:2,1:3])
# [[5 6]]
#  取第一维的索引1到索引2之间的元素，也就是第二行
# # 取第二维的索引1到索引3之间的元素，也就是第二列和第三列

#随机抽样 (numpy.random)
# rand(d0, d1, …, dn)	Random values in a given shape.
# randn(d0, d1, …, dn)	Return a sample (or samples) from the “standard normal” distribution.
# randint(low[, high, size, dtype])	Return random integers from low (inclusive) to high (exclusive).
# random_integers(low[, high, size])	Random integers of type np.int between low and high, inclusive.
# random_sample([size])	Return random floats in the half-open interval [0.0, 1.0).
# random([size])	Return random floats in the half-open interval [0.0, 1.0).
# ranf([size])	Return random floats in the half-open interval [0.0, 1.0).
# sample([size])	Return random floats in the half-open interval [0.0, 1.0).
# choice(a[, size, replace, p])	Generates a random sample from a given 1-D array
# bytes(length)	Return random bytes.


# numpy.random.rand()　生成[0, 1)间随机数
#
# numpy.random.rand(d0, d1, …, dn)函数：
# 生成一个(d0*d1* …* dn)维位于[0, 1)中随机样本
a=np.random.rand(3,2)
print(a)
# [[0.97146136 0.37404033]
#  [0.4807752  0.62259959]
#  [0.229267   0.9255095 ]]


# numpy.random.random()　
# 生成随机浮点数
#  Return random floats in the half-open interval [0.0, 1.0).
a=np.random.random()
b=np.random.random(5)
c=np.random.random(size=(3,3))
print(a)
# 0.11193873235405427
print(b)
# [0.47720402 0.08839607 0.03787762 0.91095815 0.32032341]
print(c)
# [[0.35142528 0.36613813 0.92550255]
#  [0.21280943 0.92517852 0.54451338]
#  [0.17298798 0.81640273 0.46928169]]


# numpy.random.randint()　产生随机整数
#
# API: randint(low, high=None, size=None, dtype=’l’)
# numpy.random.randint()随机生一个整数int类型，可以指定这个整数的范围
print (np.random.randint(8)) #6 不超过8
print(np.random.randint(5, size=3)) #[0 0 0]
print(np.random.randint(6, size=(3,2)))
# [[1 0]
#  [4 0]
#  [5 0]]

# numpy.random.normal() 　高斯分布随机数
# API: normal(loc=0.0, scale=1.0, size=None)
# loc：均值，scale：标准差，size：抽取样本的size

print(np.random.normal(loc=0.0, scale=1, size=(2, 3)))
#[[ 0.00528189 -0.20875828 -0.10024746]
# [-0.0626219   1.45906232  0.51213271]]

# numpy.random.randn()　标准正态分布随机数
# numpy.random.randn(d0, d1, …, dn)函数：
# 从标准正态分布中返回一个(d0*d1* …* dn)维样本值

print(np.random.randn(4, 2))
# [[ 0.28105068 -0.6257411 ]
#  [ 0.79933175  1.27780089]
#  [-1.01584908 -0.7761035 ]
#  [-1.62885819  0.59633407]]


# numpy.random.RandomState()　指定种子值
#
# numpy.random.RandomState()指定种子值（指定种子值是为了使同样的条件下每次产生的随机数一样，避免程序调试时由随机数不同而引起的问题）
# 如不设置种子值时,np.random.randint(8)可能产生0-7内的任意整数，且每次产生的数字可能是任意一种．
# 而设置种子值后,np.random.RandomState(0).randint(8)可能产生0-7内的任意整数，但种子值不变时每次运行程序产生的数字一样．

n1 = np.random.RandomState(0).random_sample()
n2 = np.random.RandomState(0).random_sample()
n3=np.random.RandomState(1).random_sample()
print(n1)#0.5488135039273248
print(n2)#0.5488135039273248
print(n3)#0.417022004702574 多次运行此执行文件，返回的值是一样的。