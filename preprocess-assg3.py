#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
import pandas as pd


# In[2]:


#input_filename = argv[1]
#output_filename = argv[2]
#sys.argv[1] = 'dating-full.csv'
#sys.argv[2] = 'dating.csv'
# Load csv
d = pd.read_csv('dating-full.csv')
d = d.head(6500)


# In[3]:


quote = 0
(row, col) = d.shape
#print row
#print col
for i in range(row):
    if d['race'][i].startswith("'") and d['race'][i].endswith("'"):
        quote += 1
    if d['race_o'][i].startswith("'") and d['race_o'][i].endswith("'"):
        quote += 1
    if d['field'][i].startswith("'") and d['field'][i].endswith("'"):
        quote += 1

d['race'] = d['race'].str.replace("'","")
d['race_o'] = d['race_o'].str.replace("'","")
d['field'] = d['field'].str.replace("'","")


# In[4]:


case = 0
for i in range(row):
    if any(letter.isupper() for letter in str(d['field'][i])):
        case += 1

d['field'] = d['field'].str.lower()


# In[5]:


preference_scores_of_participant  = ['attractive_important', 'sincere_important', 'intelligence_important', 'funny_important', 'ambition_important', 'shared_interests_important']

preference_scores_of_partner = ['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence',  'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests']

for i in range(row):
    participant_sum = 0
    partner_sum = 0

    for pref in preference_scores_of_participant:
        participant_sum += d[pref][i]
        
    for pref in preference_scores_of_partner:
        partner_sum += d[pref][i]
    
    # update the preference scores of participant
    for pref in preference_scores_of_participant:
        d.loc[i, pref] = d[pref][i]/participant_sum
        
    # update the preference scores of partner
    for pref in preference_scores_of_partner:
        d.loc[i, pref] = d[pref][i]/partner_sum


# In[6]:


categorical_attr = ['gender', 'race', 'race_o', 'field']
map_vector = {}

for attr in categorical_attr:
    map_vector[attr] = {}

#print map_vector
    
for attr in categorical_attr:
    #print d[attr].value_counts()
    l = d[attr].unique().tolist()
    l.sort()
    #print l[-1]
    
    for i in range(len(l)):
         l[i] += '_' + attr
    
    # Initialize the map vector
    for field in l:
        map_vector[attr][field] = [0 for t in range(len(l) - 1)]
    
    for i in l:
        if i != l[-1]:
            map_vector[attr][i][l.index(i)] = 1
    
    one_hot = pd.get_dummies(d[attr])
    col_name = one_hot.columns.tolist()
    #print one_hot.columns
    
    for i in range(len(col_name)):
        col_name[i] += '_' + attr

    one_hot.columns = col_name
    #print one_hot
    #pd.get_dummies(d,prefix=attr)
    d = d.drop(attr, axis=1)
    d = d.join(one_hot)
    d = d.drop(col_name[-1], axis=1)
    
    
print 'Mapped vector for female in column gender: ', map_vector['gender']['female_gender']
print 'Mapped vector for Black/African American in column race: ', map_vector['race']['Black/African American_race']
print 'Mapped vector for Other in column race_o: ', map_vector['race_o']['Other_race_o']
print 'Mapped vector for economics in column field: ', map_vector['field']['economics_field']


# In[7]:


df_test = d.sample(frac=0.2, random_state=25)
df_test.to_csv('testSet.csv', index=False)
# Subtract test from training
df_train = d[~d.index.isin(df_test.index)]
df_train.to_csv('trainingSet.csv', index=False)


# In[ ]:




