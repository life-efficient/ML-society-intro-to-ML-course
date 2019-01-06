#varibles
i = 2
f = 2.4
st = 'haron'
st1 = 'your age in 5 years is '
age = '20'
haroonis20 = True

#functions
def add(x, y):
    print(x+y)

#v = add(15, 20)

#lists
vec1 = [1, 2, 3]
vec2 = [4, 5, 6]

ls = [[1, 2, 3],
      [4, 5, 6]]

ls.append([7, 8, 9])
sls = ['my', 2, 'is', 'haron']

#dictionaries
mdict = {'000':'John', '001':'Sally', '002':'Jermaine'}

#logical operators + if statements
x=5

y=10

bo = False

if bo:
    print('bo is true')
else:
    print('bo is false')


#loops
f = 0
while f<10:
    #print(f)
    f = f+1
mylist = ['this', 'is', 'a', 'list']
#for word in mylist:
    #print(word)

#classes and objects
class car:
    def __init__(self, brand, year):
        self.brand = brand
        self.year = year
        self.state ='parked'
    def start(self):
        if self.year<=1990:
            self.state = 'stalling'
        else:
            self.state= 'going'
        print(self.state)

mycar = car('ford', 1980)
mysecondcar = car('tesla', 2016)

#mycar.start()

import numpy as np

vec = np.array([0, 0, 1])
mat = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
