'''
Created on Apr 2, 2013

@author: ben

ch05 examples
'''

#Note: do next one in ipython. 

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os

os.system('clear')

# Series 
obj1 = Series([4, 7, -5, 3])
obj2 = Series([4, 7, -5, 3], index = ['d', 'b', 'a', 'c'])
np.exp(obj2)

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(sdata)
states = ['California', 'Ohio', 'Oregon', "Texas"]
obj4 = Series(sdata, index=states)
obj4.name = 'population'
obj4.index.name = 'state'

obj1.index = ['Bob', 'Steve', 'Jeff', 'Ryan']


# DataFrame
data = {'state' : ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year' : [2000, 2001, 2002, 2001, 2002],
        'pop' : [1.5, 1.7, 3.6, 2.4, 2.9]}

frame = DataFrame(data)
frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'], 
                                  index=['one', 'two', 'three', 'four', 'five'])
frame2['state']
frame2.values
frame2.year
frame2.ix['three']
frame2.ix[3]

frame2['debt'] = 16.5
#frame2['debt'] = [16.5, 17, 26]
frame2.debt = np.arange(5.)
val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2.debt = val
frame2['eastern'] = frame2.state =='Ohio'
del frame2['eastern']

# dict of dicts into DataFrame
pop = {'Nevada': {2001: 2.5, 2002: 2.9}, 'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = DataFrame(pop)
frame3.index.name = 'year'; frame3.columns.name = 'state'

frameTest = DataFrame(frame3)
frameTest = DataFrame(frame3, index=[1,2,3])

# pandas Index object

obj1 = Series(range(3), index=['a', 'b', 'c'])
index = obj1.index
index = pd.Index(range(3))
obj2 = Series([1.5, -2.5, 0], index=index)
#obj2 is obj2
#obj2.index is index
obj3 = obj2.copy()
# obj3 is obj2 # evaluates to FALSE - so is compares pointers and not the actual data. 

# Some python functions.

obj1 = Series([4.5, 7.2, -5.3, 3.7], index = ['d', 'b', 'a', 'c'])
obj2 = obj1.reindex(['a', 'b', 'c', 'd', 'e'])
obj2 = obj1.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0)

frame = DataFrame(np.arange(9).reshape((3,3)), index=['a', 'c', 'd'], 
                  columns=['Ohio', 'Texas', 'California'])
frame2 = frame.reindex(['a', 'b', 'c', 'd'])
states = ['Texas', 'Utah', 'California']
frame3 = frame.reindex(columns=states)

#This one is cool!!
frame4 = frame.reindex(index=['a', 'b', 'c', 'd'], column =states)
frame5 = frame.ix[['a', 'b', 'c', 'd'], states]

# Indexing 
obj1 = Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
#try: obj1[[1]], obj1[1]], obj1[['b', 'a', 'd']], obj1['b':'c'], obj1['b':'elephant']
data = DataFrame(np.arange(16).reshape((4,4)), 
                 index=['Ohio', 'Colorado', 'Utah', 'New York'], 
                 columns=['one', 'two', 'three', 'four'])

# Data Alignment
s1 = Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])

#broadcasting examples.
arr = np.arange(12.).reshape((3,4))
test_arr = arr-arr[0]

frame = DataFrame(np.arange(12.).reshape((4,3)), columns =list('bde'), 
                  index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.ix[0]
bcast_test1 = frame - series
bcast_test2 = series - frame 

# apply: kind of like R. 
f = lambda x: x.max() - x.min()
frame = DataFrame(np.random.randn(4,3), columns=list('bde'),
                  index=['Utah', 'Ohio', 'Texas', 'Oregon'])
#try frame.apply(f), frame.apply(f, axis=1)
formatter = lambda x:'%.2f' % x
frame.applymap(formatter)

# check out assignment. 
frametest = frame
frametest['e'] = frametest['e'].map(formatter)
# now look at frame. 

# Basic statistics over frames

df = DataFrame([[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [0.75, -1.3]], 
               index=['a', 'b', 'c', 'd'], 
               columns=['one', 'two'])
#df.sum(), df.sum(axis=1)
df.mean(axis=1)
df.mean(axis=1, skipna=False)
df.describe() #cool
df.describe()['one'] 
test = df.describe()

# Finance data
import pandas.io.data as web
all_data = {}
for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOG']:
    all_data[ticker] = web.get_data_yahoo(ticker, '1/1/2000', '1/1/2010')