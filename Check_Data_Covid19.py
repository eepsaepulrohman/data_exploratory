#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats
from mlxtend.preprocessing import minmax_scaling
import seaborn as sns
import matplotlib.pyplot as plt

# membaca semua data

data = pd.read_csv("covid_19_indonesia_time_series_all.csv")

# set seed for reproducibilty
np.random.seed(0)


# In[2]:


#Sekarang kita bisa melihat data yang hilang
data.head()


# In[3]:


#melihat data jumlah data yang hilang per kolom
missing_values_count = data.isnull().sum()

#sekarang lihat poin yang hilang di sepuluh kolom pertama
missing_values_count[0:10]


# In[4]:


# berapa banyak total nilai yang hilang
total_cells = np.product(data.shape)
total_missing = missing_values_count.sum()

#persen data yang hilang
percent_missing = (total_missing/total_cells) * 100
print('Persen Data Hilang:',percent_missing)
missing_values_count[0:10]


# In[5]:


#hapus semua baris yang berisi nilai yang telah hilang
data.dropna()


# In[6]:


#hapus semua kolom dengan setidaknya satu nilai yang hilang

columns_with_na_dropped = data.dropna(axis=1)
columns_with_na_dropped.head()


# In[7]:


#Compare data origin dengan data yang telah di hapus

print("Columns in original dataset: %d \n" % data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])


# In[8]:


data.fillna(0)


# In[9]:


data.fillna(method='bfill', axis=0).fillna(0)


# In[10]:


# menghasilkan 1000 titik data yang diambil secara acak dari distribusi eksponensial
data = np.random.exponential(size=1000)

# mix-max skala data antara 0 dan 1
scaled_data = minmax_scaling(data, columns=[0])

# membandingkan
fig, ax = plt.subplots(1,2)
sns.distplot(data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")


# In[13]:


# menormalisasi data eksponensial dengan boxcox
normalisasi_data = stats.boxcox(data)

# membandingkan
fig, ax=plt.subplots(1,2)
sns.distplot(data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalisasi_data[0], ax=ax[1])
ax[1].set_title("Normalisasi Data")


# In[ ]:




