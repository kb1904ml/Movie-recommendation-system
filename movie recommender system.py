#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


movies=pd.read_csv(r"C:\Users\user\Downloads\archive (11)\tmdb_5000_movies.csv")


# In[3]:


credits=pd.read_csv(r"C:\Users\user\Downloads\archive (11)\tmdb_5000_credits.csv")


# In[4]:


movies.head()


# In[5]:


credits.head(1)


# In[6]:


credits.head(1)['cast'].values


# In[7]:


movies=movies.merge(credits,on='title')


# In[8]:


movies.head()


# useful columns
# 1.genres
# 2.id
# 3.keywords
# 4.title
# 5.overview
# 6.cast
# 7.crew

# In[9]:


movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[10]:


movies.head()


# In[11]:


movies.isnull().sum()


# In[12]:


movies.dropna(inplace=True)


# In[13]:


movies.isnull().sum()


# In[14]:


movies.duplicated().sum()


# In[15]:


movies.iloc[0].genres


# The above output is in form of dictionary and we have to convert in the form as ['Action','Adventure','Fantasy','SciFi']
# 

# In[16]:


import ast


# In[17]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[19]:


movies['genres']=movies['genres'].apply(convert)


# In[20]:


movies.head()


# In[21]:


movies['keywords']=movies['keywords'].apply(convert)


# In[22]:


movies.head()


# In[23]:


movies['cast'][0]
#here from each dictionary we only need name key and we need names of top three cast members


# In[24]:


def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter !=3:
             L.append(i['name'])
             counter+=1
        else:
            break
    return L
            


# In[25]:


movies['cast']=movies['cast'].apply(convert3)


# In[26]:


movies['crew'][0]
#our area of interest is only that key where job is direction


# In[27]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L
        
        
     


# In[28]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[29]:


movies.head()


# In[30]:


# overview column is in string format so we are converting it into list format so that we can easily concat it with other columns


# In[31]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[32]:


# now we are removing the spaces between the names 


# In[33]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[34]:


movies.head()


# In[35]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[36]:


movies.head()


# In[37]:


movies_df=movies[['movie_id','title','tags']]


# In[38]:


movies_df


# In[39]:


movies_df['tags']=movies_df['tags'].apply(lambda x:" ".join(x))


# In[40]:


movies_df.head()


# In[48]:


import nltk


# In[49]:


#stemming
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()


# In[52]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[55]:


movies_df['tags']=movies_df['tags'].apply(stem)


# In[56]:


movies_df['tags'][0]


# In[57]:


movies_df['tags']=movies_df['tags'].apply(lambda x:x.lower())


# In[58]:


movies_df.head()


# In[59]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[60]:


vectors=cv.fit_transform(movies_df['tags']).toarray()


# In[61]:


vectors


# In[62]:


#get_feature_names() displays most occuring 5000 words
cv.get_feature_names()


# In[50]:


#stemming
ps.stem('loved')


# In[51]:


ps.stem('dance')


# In[63]:


from sklearn.metrics.pairwise import cosine_similarity


# In[66]:


similarity=cosine_similarity(vectors)


# In[64]:


cosine_similarity(vectors).shape


# In[71]:


similarity[0]


# In[73]:


#we want to sort the movies similarities from index 1 as index 0 will be the movie itself
sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[78]:


def recommend(movie):
    movie_index=movies_df[movies_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(movies_df.iloc[i[0]].title)
    


# In[80]:


recommend('Batman Begins')


# In[81]:


movies_df.iloc[1216].title

