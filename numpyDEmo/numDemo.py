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



