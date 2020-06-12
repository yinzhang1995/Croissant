# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 17:12:01 2017

@author: anel
"""
#Week 1###############################
myString='Hello World!'
print(myString)

price=input('input the price of an apple:')
type(price)

price=int(input('input the price of an apple:'))
type(price)

if (signal=='red') and (car=='moving'):
    car='stop'
elif (signal=='green') and (car=='stop'):
    car='moving'
    
x='today'
y='is'
z='Sunday'
print(x,y,z)

p=3.14
myString='is the mathematic circular constant'
print(p,myString)

#查看关键字
import keyword
print(keyword.kwlist)

#判断两个数是否相等  字符串不能用
p=3
q=3
p is q 
#返回True

#连续赋值
PI=pi=3.14

#多重赋值
x=1
y=2
x,y

x,y=y,x  #交换

temp=3.14,3
PI,r=temp
#得到PI=3.14 r=3

9.8e3
-4.78e-1

x=2.4+5.6j
type(x)
x.imag
x.real
x.conjugate

#序列包括 字符串（不可变 ''）、列表（可变 []）、元组（不可变 ()）
#映射类型：字典  键值对
d={'sine':'sin','cosine':'cos','PI':3.14159}
d['sine']

not(x<5.0)
(x<5.0) and (y>2.718)
(x<5.0) or (y>2.718)
not (x is y)

dir(__builtins__)  #查找内建函数

#Week 2###############################
import math
math.e
math.pi
math.ceil
math.pow
math.log
math.sqrt
math.degrees  #弧度转角度
math.radians  #角度转弧度

import os
os.getcwd()  #获得当前路径
os.chdir(newdir)  #转换路径
os.rename('fundamental.py', 'fundamental1.py')   #重命名一个文件，但文件不能在打开状态
os.mkdir('C:\\Users\\anel\\tt')  #创建一个新路径
os.rmdir('C:\\Users\\anel\\tt')  #删除一个目录

import random
random.choice(['c++','c','java'])
random.randint(a,b)  #Return random integer in range [a, b], including both end points.
random.randrange(0,10,2) #range(0,10,2)中随机取一个值
random.random()  #random() -> x in the interval [0, 1)
random.uniform(5,10)  #Get a random number in the range [a, b) or [a, b] depending on rounding
random.sample(range(100),10)  #随机取样
random.shuffle(list)  #参数必须是一个列表名！！！ 不能是类似list(range(10))
#如果是list(range(10))形式 则返回值为none
#就像传递进去一个引用

import datetime
from datetime import date  #可以直接写date() 而不需要datetime.time()
from datetime import time
tm=time(23,20,35)  #就像构造函数一样初始化一个类
print(tm)  #23:20:35 直接看tm只是一个类
from datetime import datetime
dt=datetime.now()
print(dt.strftime('%a,%b %d %Y %H:%M')) #Wed,Jun 14 2017 18:38
dt=datetime(2017,2,3,23,29)
ts=dt.timestamp()  #时间转换为时间戳
print(datetime.fromtimestamp(ts)) #时间戳转换为时间
#实际上dt就是一个时间类dt.strftime实际上就是类中封装的函数

list(range(0,10,2))  #返回列表
range(0,10,2)  #只返回range类 range(0,10,2)

while ....:
else:  #不满足循环条件做什么
    
for iter_var in iterable_object:
#iterable_object 可以是 string list tuple dictionary file
for i in range(3,11,2):
    print(i,end=' ')

print('i={},sum={}'.format(i,sumA))
print('{:d} is not a prime.'.format(2))
#2 is not a prime.
print('{:f} is not a prime.'.format(2))
#2.000000 is not a prime.


def addMe2Me(x):
    return(x+x)

def addMe2Me(x,y=1):   #默认参数 
#默认参数一般需要放置在参数列表的最后

r=lambda x:x+x      #r(5)
r=lambda x,y:x*x+y   #r(2,3)
#r相当于一个函数句柄   

def f(x):
    global a     #强调a为全局变量  
    print(a)
    a = 5
    print(a + x)
#调用
a=3   #若没有global a 这句 a=3会报错
f(8)

#查看异常类
dir(__builtins__)

try:
    
except ValueError:

except ZeroDivisionError:
    
#-------
try:

except(ValueError,ZeroDivisionError):

#-------
try:

except ValueError as err:
    print(err)
#-------
try:
    
except:

else:   #如果没有出错
#-------

f=open('data.txt')
for line in f:
    print(line,end=' ')

with open('data.txt') as f:
    for line in f:
        print(line,end='')   
        
#week 3#################################
#序列：字符串、列表、元组
#对象身份比较 is / is not
#把一个列表倒置
week[::-1]
#判断是否属于
'BA' in ('BA','The Boeing Company','184.76')
'aa' in ['aa','bb','cc']

list('Hello World!')  #转换为列表 ['H', 'e', 'l', 'l', 'o', ',', ' ', 'W', 'o', 'r', 'l', 'd', '!']
tuple('Hello WOrld!')  #转化为元组 ('H', 'e', 'l', 'l', 'o', ',', ' ', 'W', 'o', 'r', 'l', 'd', '!')
sorted(aStr)  #排序

print('There are {1:d} punctuation marks. '.format(200,300))
# 1表示format括号里的数字序号

<左对齐
>右对齐
^居中对齐
#把一个字符串反过来并输出
sStr=''.join(reversed(sStr))

str1 = "this is string example....wow!!!";
str2 = "exam";
 
print str1.index(str2);
print str1.index(str2, 10);  #str2首先出现在str1中的位置
print str1.rindex(str2, 10);  #str2最后出现在str1中的位置
#10位开始索引的位置

jScores=[1,3,2,7,5]
jScores.sort()
jScores.pop()   #弹出最后一个
jScores.pop(0)  #弹出第一个
jScores.append(10)


week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
weekend = ['Saturday', 'Sunday']
week.extend(weekend)
Out[30]: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

week.append(weekend)
Out[32]: 
['Monday',
 'Tuesday',
 'Wednesday',
 'Thursday',
 'Friday',
 'Saturday',
 'Sunday',
 ['Saturday', 'Sunday']]

for i,j in enumerate(week):
    print(i+1, j)
    
1 Monday
2 Tuesday
3 Wednesday
4 Thursday
5 Friday

[x*x for x in range(10)]
[(x+1, y+1) for x in range(2) for y in range(2)]

#可变长参数
def foo(args1, *argst):
print(args1)
print(argst)

foo('Hello,', 'Wangdachui', 'Niuyun', 'Linling')
Hello,
('Wangdachui', 'Niuyun', 'Linling')

def foo():
    return 1, 2, 3

foo()
(1, 2, 3)

#week 4#######################
#查找下标
names = ['Wangdachui', 'Niuyun', 'Linling', 'Tianqi']
names.index('Niuyun')

names = ['Wangdachui', 'Niuyun', 'Linling', 'Tianqi']
>>> salaries = [3000, 2000, 4500, 8000]
>>> dict(zip(names,salaries))
{'Tianqi': 8000, 'Wangdachui': 3000, 'Niuyun': 2000, 'Linling': 4500}

#让值全部为3000
aDict= {}.fromkeys(('Wangdachui', 'Niuyun', 'Linling', 'Tianqi'),3000)

aInfo.keys()
aInfo.values()
#字典的更新
aInfo.update(bInfo) #bInfo 也是一个字典

stock['AAA'] #查找  如果没有会报错
stock.get('AAA')  #查找 如果没有不会报错

aStock={} #清空字典
aStock.clear()

#字典作为参数
def func(args1, *argst, **argsd):
    print(args1)
    print(argst)
    print(argsd)
>>> func('Hello,','Wangdachui','Niuyun','Linling',a1= 1,a2=2,a3=3)
Hello,
('Wangdachui', 'Niuyun', 'Linling')
{'a1': 1, 'a3': 3, 'a2': 2}

#集合
nameSet=set(names)  #形成集合
aSet = set('hello')  #可变集合
bSet = frozenset('hello') #不可变集合
'u' in aSet
aSet & bSet #交
aSet | bSet #并
aSet ^ bSet #差  减掉共同含有的元素
aSet.issubset(bSet)  #aSet是否是bSet的子集
aSet.intersection(bSet)
aSet.difference(bSet)
aSet('!')
aSet.remove('!')
aSet.update('Yeah')

#SciPy
    #ndarray N维数组
    #Series  变长字典
    #DataFrame 数据框
    
    #NumPy
import numpy as np
from scipy import linalg
arr=np.array([[1,2],[3,4]])
linalg.det(arr)

ndaaray.ndim()
ndarray.shape()
adarray.size()
ndarray.dtype()
ndarray.itemsize()

import numpy as np
aArray=np.array([1,2,3])
aArray=np.array([[1,2,3],[4,5,6]])
#array([[1, 2, 3],
#       [4, 5, 6]])
np.arange(1,5,0.5)
np.random.random((2,2))
np.linspace(1,2,10,endpoint=False)
np.ones([2,3])
np.zeros([2,3])
np.fromfunction(lambda i,j:(i+1)*(j+1),(9,9))

aArray[1]  #取第1行
aArray[0:2] #取0,1行 不包括2
aArray[:,[0,1]]

aArray.reshape(3,2)
aArray.resize(3,2)

np.vstack((aArray,bArray))
np.hstack((aArray,bArray))

#*\+- 都是对应元素运算

aArray.sum(axis=0) #最后是一个行向量
aArray.sum(axis=1)
aArray.argmax() #返回最大元素的下标

np.linalg.det(x)
np.linalg.inv(x)
np.dot(x,x)
np.power(a,2)

#计时 
import time
time.clock()
#ufunc（universal function）是一种能对数组的每个元素进行操作的函数。

#Series 由数据和索引组成
from pandas import Series
aSer=pd.Series([1,2.0,'a'])
bSer= pd.Series(['apple','peach','lemon'], index = [1,2,3])
bSer.index
bSer.values
aSer = Series([3,5,7],index = ['a','b','c'])
#用索引找数据
aSer['b']
np.exp(aSer)  #求Series aSer中数据的e指数
#Series的数据对齐'
data = {'AXP':'86.40','CSCO':'122.64','BA':'99.44'}
sindex = ['AXP','CSCO','BA','AAPL']
>>> aSer = pd.Series(data, index = sindex)
>>> aSer
AXP 86.40
CSCO 122.64
BA 99.44
AAPL NaN
dtype: object

>>> pd.isnull(aSer)
AXP False
CSCO False
BA False
AAPL True
dtype: bool

>>> bSer= {'AXP':'86.40','CSCO':'122.64','CVX':'23.78'}
>>> cSer= pd.Series(bSer)
>>> aSer + cSer
AAPL NaN
AXP 86.4086.40
BA NaN
CSCO 122.64122.64
CVX NaN
dtype: object

aSer.index.name = 'volume'

#DataFrame   大致可看成共享同一个index的Series集合
data = {'name': ['Wangdachui', 'Linling', 'Niuyun'], 'pay': [4000, 5000, 6000]}
frame = pd.DataFrame(data)

data = np.array([('Wangdachui', 4000), ('Linling', 5000), ('Niuyun', 6000)])
frame =pd.DataFrame(data, index = range(1, 4), columns = ['name', 'pay'])

frame.index
frame.columns
frame.values

frame['name']
frame.name
frame.iloc[:2,1]
frame['name']='admin'  #name列全体赋值为admin
del frame['pay']
frame.pay.min()
frame[frame.pay>='5000']    #注意 是字符串‘5000’
           
import pandas as pd
quotesdf=pd.read_csv('axp.csv')

quotesdf=pd.DataFrame(quotes)
quotesdf.index=range(1,len(quotes)+1)

from datetime import date
firstday=date.fromtimestamp(1464010200)
lastday=date.fromtimestamp(1495200600)
firstday   #datetime.date(2016, 5, 23)

y=date.strftime(x,'%Y-%m-%d')
quotesdf.drop(['date'],axis=1)  #删除date列

#显示行索引
list(djidf.index)
#显示列索引
list(djidf.columns)

djidf.head(5)
djidf.tail(5)

quotesdf['2017-05-01':'2017-05-05']   #中括号里的是行index
djidf.loc[1:5,]  等价于 djidf.iloc[1:6,]
djidf.loc[:,['code','lasttrade']]
djidf.at[1,'lasttrade']
djidf.iat[1,2]

quotesdf[(quotesdf.index>= '2017-03-01')& (quotesdf.index<= '2017-03-31')]

djidf[djidf.lasttrade>= 180].name
status = np.sign(np.diff(quotesdf.close))
status[np.where( status == 1.)].size

djidf.sort_values(by = 'lasttrade', ascending = False)

#计数统计
tempdf['month'].value_counts()

temp.tm_mon  #取日期的月份

tempdf.groupby('month').count()

tempdf.groupby('month').sum().volume

pieces = [tempdf[:5], tempdf[len(tempdf)-5:]]
pd.concat(pieces)

pd.merge(djidf.drop(['lasttrade'], axis = 1), AKdf, on = 'code')
#on 是按照什么连接
#找到两个表中相同的然后合并起来 
http://matplotlib.org/gallery.html

plt.savefig('1.jpg')
plt.plot(t,t,t,t+2,t,t**2)
plt.plot(x,y,'o')  #散点图
plt.bar(x,y)

import pandas as pd
closeMeanKO.plot()
quotesdfIBM.close.plot()

retrieve_quotes_historical('IBM')

quotesIIdf=pd.DataFrame() #创建一个新的dataframe
quotesIIdf['IBM']=IBM_volumn #把数据放进去
quotesIIdf.plot(kind='bar') #柱状图
quotesIIdf.plot(kind='hbar') #横向柱状图
quotesIIdf.plot(kind='bar',stacked=True) #堆积效果的柱状图
quotesINTC.plot(kind='pie',subplots=True,autopct='%.2f') #饼图、并设置饼图上的数字格式
quotesIIdf.plot(market='v')
quotesIIdf.boxplot()   #箱线图（盒须图）

#箱线图：数据分散情况、中位数四分之一分位点四分之三分位点比较可以看出数据对称情况

df=pd.DataFrame(quotes)
df.to_csv('stock.csv')  #写csv函数
result=pd.read_csv('stockAXP.csv')
result['close']

df.to_excel('stockAXP.xlsx',sheet_name='AXP')

import pandas as pd
stu_df = pd.DataFrame()
stu_df = pd.read_excel('stu_scores.xlsx', sheet_name = 'scores')
stu_df['sum'] = stu_df['Python'] + stu_df['Math']
stu_df.to_excel('stu_scores.xlsx', sheet_name = 'scores')

from nltk.corpus import gutenberg
allwords=gutenberg.words('shakespeare-hamlet.txt')
len(allwords)  #总字数
len(set(allwords))  #不重复字数 利用集合处理
allwords.count('Hamlet')  #计数'Hamlet'的数量
longwords=[w for w in A if len(w)>12]

#GUI与面向对象#########################
class ClassName(object):
    'define ClassName class'
    class_suite
    
class MyData(object):
    'this is a very simple example class'
    pass

class Dog(object):
    def greet(self):
        print('Hi!')
        
dog=Dog()
dog.greet()

def _init_(self,...):  #相当于构造函数

#基类和派生类
class SubClassName(ParentClass1,):
    'define ClassName class'
    class_suite
        
def timeConversion(s):
    # Complete this function
    if s[-2:]=='AM':
        if s[:2]=='12':
            result='00:00:00'
        else:
            hour=int(s[0:2])%12
            result= str(hour)+s[2:-2]
    if s[-2:]=='PM':
        if s[0:2]=='12':
            result='12:00:00'
        else:
            hour=int(s[0:2])+12
            result=str(hour)+s[2:-2]
    return(result)
        
s = raw_input().strip()
result = timeConversion(s)
print(result)

######
def BuildHeap(freq):
  for i in range(int(int(N)/2),0,-1):
    x = freq[i]
    j = i
    while j < len(freq):
      j = i * 2 + 1
      child = j
      if child < int(N) - 1 and freq[child][1] > freq[child + 1][1]:
        child = child + 1
      if freq[child][1] < x[1]:
        break
      else:
        freq[j][1] = freq[child][1]
      
      i = child
    freq[j] = x
  return freq


######
class HTree:
  def __init__(self, cargo, left=None, right=None):
      self.cargo = cargo
      self.left  = left
      self.right = right

def BuildHeap(freq):
  for i in range(int((len(freq) - 2)/2),-1,-1):
    x = freq[i]
    j = i * 2 + 1
    child = i
    temp = i
    while j <= len(freq) - 1:
      child = j
      if child + 1 <= len(freq) - 1 and freq[child][1] > freq[child + 1][1]:
        child = child + 1
      if freq[child][1] > x[1]:
        break
      else:
        freq[temp] = freq[child]
      temp = child
      j = temp * 2 + 1
    
    freq[temp] = x
  return freq

def BuildHTree(freq):
    
    while len(freq) >= 2:
        one = HTree(freq[0])
        freq = BuildHeap(freq[1:])
        two = HTree(freq[0])
        freq = BuildHeap(freq[1:])
        three = HTree(str(int(one.cargo[1]) + int(two.cargo[1])))
        three.left = one.cargo[0]
        three.right = two.cargo[0]
        
        freq.append([three,three.cargo])
        freq = BuildHeap(freq)
    
    return three

def getlevel(root):
    if root in key:
        return 0
    if root.left in key and root.right not in key:
        return getlevel(root.right) + 1
    if root.right in key and root.left not in key:
        return getlevel(root.left) + 1
    
    return max(getlevel(root.left),getlevel(root.right))

def calculate(root,depth):
    
    if root in key:
        return depth * int(dict1[root])
    else:
        return calculate(root.left,depth + 1) + calculate(root.right,depth + 1)
    
    
def checktestdata(testdata,length):
# first check length
  templen = 0
  for i in range(len(testdata)):
    templen = templen + int(dict1[testdata[i][0]]) * len(testdata[i][1])
  if templen != length:
      return 0
  
  # check for pre-code
  t = HTree(0)
  head = t
  for i in range(len(testdata)):
      count = 0
      t = head
      for j in testdata[i][1]:
          if int(j) == 0 and count < len(testdata[i][1]) - 1 and t.left == None:
              t.left = HTree(0)
              t = t.left
          
          elif int(j) == 0 and count < len(testdata[i][1]) - 1 and t.left.cargo == 0:
              t = t.left
          
          elif int(j) == 1 and count < len(testdata[i][1]) - 1 and t.right == None:
              t.right = HTree(0)
              t = t.right
              
          elif int(j) == 1 and count < len(testdata[i][1]) - 1 and t.right == 0:
              t = t.right
              
          elif int(j) == 0 and count == len(testdata[i][1]) - 1 and t.left == None:
              t.left = HTree(testdata[i][0])
              t = t.left
              
          #elif int(j) == 0 and count == len(testdata[i][1]) - 1 and t.left == 0:
              #t.left = HTree(testdata[i][0])
              #t = t.left
              
          elif int(j) == 1 and count == len(testdata[i][1]) - 1 and t.right == None:
              t.right = HTree(testdata[i][0])
              t = t.right
              
          #elif int(j) == 1 and count == len(testdata[i][1]) - 1 and t.right == 0:
              #t.right = HTree(testdata[i][0])
              #t = t.right
              
          else:
              #return 0
              k = 0
              
          count = count + 1
  
  return 1         
      
globals N
globals key
globals dict1
N = input()
N = int(N)
freq = input().split(' ')
key = freq[0::2]
value = freq[1::2]
dict1 = dict(zip(key,value))

freq_temp = []
for i in range(len(key)):
    freq_temp.append([key[i],value[i]])
freq = freq_temp


freq = BuildHeap(freq)
root = BuildHTree(freq)
#l = getlevel(root)
length = calculate(root,0)

testnum = input()
testnum = int(testnum)
for i in range(testnum):
  testdata = []
  for j in range(len(freq)):
    n,code = input().split(' ')
    testdata.append([n,code])
  
  result = checktestdata(testdata,length)
  if result:
    print('Yes')
  else:
    print('No')
  
    

A 1 B 1 C 1 D 3 E 3 F 6 G 6

A 00000
B 00001
C 0001
D 001
E 01
F 10
G 11

