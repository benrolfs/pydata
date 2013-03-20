'''
Created on Mar 6, 2013

@author: ben
'''
import os, json
os.getcwd()
os.chdir('/home/ben/eclipse/pydata')
path = 'ch02/usagov_bitly_data2012-03-16-1331923249.txt'
open(path).readline()


records = [json.loads(line) for line in open(path)]
records[0]
records[0]['tz']

time_zones = [rec['tz'] for rec in records if 'tz' in rec]

# count records by iterating over a dict:
def get_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] +=1
        else:
            counts[x] = 1
    return counts

counts = get_counts(time_zones)

# count using collections
from collections import Counter
counts = Counter(time_zones)
counts.most_common(10)

# count using pandas
from pandas import DataFrame, Series
import pandas as pd
import numpy as np
frame = DataFrame(records)

import matplotlib.pyplot as plt
tz_counts=frame['tz'].value_counts()
clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
tz_counts = clean_tz.value_counts()
tz_counts[:10].plot(kind='barh', rot = 0)
plt.show()


results = Series([x.split()[0] for x in frame.a.dropna()])
results.value_counts()

#get operating systems
cframe = frame[frame.a.notnull()]
operating_system = np.where(cframe['a'].str.contains('Windows'), 'Windows', 'Not Windows')

#group data by tz and os
by_tz_os = cframe.groupby(['tz', operating_system])
agg_counts = by_tz_os.size().unstack().fillna(0)

indexer = agg_counts.sum(1).argsort()
count_subset = agg_counts.take(indexer)[-10:]
count_subset.plot(kind='barh', stacked=True)
plt.show()
normed_subset = count_subset.div(count_subset.sum(1), axis=0)
normed_subset.plot(kind='barh', stacked=True)
plt.show()

### MovieLens DataSet ###
unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table('ch02/movielens/users.dat', sep='::', header=None, names=unames)

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('ch02/movielens/ratings.dat', sep='::', header=None, names=rnames)

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('ch02/movielens/movies.dat', sep='::', header=None, names=mnames)

data = pd.merge(pd.merge(ratings, users), movies)

mean_ratings = data.pivot_table('rating', rows='title', cols='gender', aggfunc='mean')

ratings_by_title = data.groupby('title').size()
active_titles = ratings_by_title.index[ratings_by_title >= 250]

mean_ratings = mean_ratings.ix[active_titles]

top_female_ratings = mean_ratings.sort_index(by='F', ascending=False)

mean_ratings['Diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_index(by='Diff', ascending=True)

# if we want to reverse order of rows:
sorted_by_diff[::-1][:15]

#get stds
rating_std_by_title = data.groupby('title')['rating'].std()
rating_std_by_title.order(ascending=False)[:10]

### Baby Names ###
names1880 = pd.read_csv('ch02/names/yob1880.txt', names=['name','sex','births'])
years = range(1880, 2011)
pieces = []
columns =['name','sex','births']
for year in years:
    path = 'ch02/names/yob%d.txt' % year
    frame = pd.read_csv(path, names=columns)
    frame['year'] = year
    pieces.append(frame)
    
names = pd.concat(pieces, ignore_index=True)

total_births = names.pivot_table('births', rows ='year', cols='sex', aggfunc='sum')
total_births.plot(title='Total birhts by sex and year')
plt.show()

def add_prop(group):
    births = group.births.astype(float)
    group['prop'] = births/births.sum()
    return group
names = names.groupby(['year', 'sex']).apply(add_prop)


# check if all sums are close to 1, in the prop column
np.allclose(names.groupby(['year','sex']).prop.sum(),1)
names.groupby(['year','sex']).prop.sum()

def get_top1000(group):
    return group.sort_index(by='births', ascending=False)[:1000]            

grouped = names.groupby(['year', 'sex'])
top1000 = grouped.apply(get_top1000)
boys = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']
total_births = top1000.pivot_table('births', rows='year', cols='name', aggfunc=sum)

subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]
subset.plot(subplots=True, figsize=(12, 10), grid=False, title="Number of births per year")

table = top1000.pivot_table('prop', rows='year', cols='sex', aggfunc=sum)
table.plot(title='Sum of table1000.prop by year and sex', yticks=np.linspace(0,1.2,13), xticks=range(1880,2020,10))


# now we want to get how many popular names to reach 50% proportion (to see name diversity)
df = boys[boys.year==2010]
prop_cumsum = df.sort_index(by='prop', ascending=False).prop.cumsum()
prop_cumsum.searchsorted(0.5)

def get_quantile_count(group, q=0.5):
    group = group.sort_index(by='prop', ascending=False)
    return group.prop.cumsum().searchsorted(q)+1

diversity = top1000.groupby(['year', 'sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')
diversity.plot(title='Number of popular names in top 50% of births')

# Explore the "last letter revolution"

get_last_letter = lambda x : x[-1]
last_letters = names.name.map(get_last_letter)
last_letters.name = 'last_letter'
table = names.pivot_table('births', rows=last_letters, cols=['sex', 'year'], aggfunc=sum)
subtable = table.reindex(columns=[1910, 1960, 2010], level='year')
letter_prop = subtable /subtable.sum().astype(float)
fig, axes = plt.subplots(2, 1, figsize=(10,8))
letter_prop['M'].plot(kind='bar', rot=0, ax=axes[0], title='Male')
letter_prop['F'].plot(kind='bar', rot=0, ax=axes[1], title='Female', legend=False)

#make time series of certain letters
letter_prop = table/table.sum().astype(float)
dny_ts = letter_prop.ix[['d', 'n', 'y'], 'M'].T