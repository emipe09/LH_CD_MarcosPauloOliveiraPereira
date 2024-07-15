import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

from collections import Counter
import nltk
import string

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer


# Carregando o dataframe
df = pd.read_csv('desafio_indicium_imdb.csv')
df['Gross'] = df['Gross'].str.replace(',', '')
df['Gross'] = pd.to_numeric(df['Gross'])

# Verificando valores faltantes
df.isna().sum()
df.shape

# Diretores que mais aparecem na lista

df['Director'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 diretores que mais aparecem na lista')
plt.ylabel('Quantidade de filmes')
plt.xlabel('Diretores')
plt.show()

# Atores que mais aparecem na lista (considerando Star1, Star2, Star3 e Star4)


df[['Star1', 'Star2', 'Star3', 'Star4']].stack().value_counts().head(10).plot(kind='bar')
plt.title('Top 10 atores que mais aparecem na lista')
plt.ylabel('Quantidade de filmes')
plt.xlabel('Atores')
plt.show()

diretores = df['Director'].value_counts()
atores = df[['Star1', 'Star2', 'Star3', 'Star4']].stack().value_counts()


filtred_diretores = diretores[diretores > 2].index
filtered_diretores = df[df['Director'].isin(filtred_diretores)]

mean_imdb_by_director = filtered_diretores.groupby('Director')['IMDB_Rating'].mean().sort_values(ascending=False).head(20)

# Plotar o gráfico
mean_imdb_by_director.plot(kind='bar')
plt.title('Diretoes com melhor média de notas IMDB (mais que 2 filmes na lista)' )
plt.ylabel('Nota IMDB')
plt.xlabel('Diretores')
plt.show()

# Atores com melhor média de notas IMDB (mais que 2 filmes na lista)
filtred_atores = atores[atores > 2].index
filtered_atores = df[df[['Star1', 'Star2', 'Star3', 'Star4']].isin(filtred_atores).any(axis=1)]

mean_imdb_by_ator = filtered_atores.groupby('Star1')['IMDB_Rating'].mean().sort_values(ascending=False).head(20)    

# Plotar o gráfico
mean_imdb_by_ator.plot(kind='bar')
plt.title('Atores com melhor média de notas IMDB (mais que 2 filmes na lista)' )
plt.ylabel('Nota IMDB')
plt.xlabel('Atores')
plt.show()

# Fazendo o mesmo para nota Meta Score

mean_meta_by_director = filtered_diretores.groupby('Director')['Meta_score'].mean().sort_values(ascending=False).head(20)

#Plotar gráfico

mean_meta_by_director.plot(kind='bar')
plt.title('Diretores com melhor média de notas Meta Score (mais que 2 filmes na lista)' )
plt.ylabel('Nota Meta Score')
plt.xlabel('Diretores')
plt.show()

mean_meta_by_ator = filtered_atores.groupby('Star1')['Meta_score'].mean().sort_values(ascending=False).head(20)    

# Plotar o gráfico
mean_meta_by_ator.plot(kind='bar')
plt.title('Atores com melhor média de notas Meta Score (mais que 2 filmes na lista)' )
plt.ylabel('Nota Meta Score')
plt.xlabel('Atores')

# Filmes com maior arrecadação

df[['Series_Title', 'Gross']].sort_values('Gross', ascending=False).head(10).plot(kind='bar', x='Series_Title', y='Gross')
plt.title('Top 10 filmes com maior arrecadação')
plt.ylabel('Arrecadação')
plt.xlabel('Filmes')
plt.show()

# Diretores com o melhor histórico de faturamento

mean_gross_by_director = filtered_diretores.groupby('Director')['Gross'].mean().sort_values(ascending=False).head(20)
mean_gross_by_director.plot(kind='barh', figsize=(10, 10), title
='Diretores com o melhor histórico de faturamento')

plt.title('Diretores com o melhor histórico de faturamento')
plt.ylabel('Diretores')
plt.xlabel('Faturamento')
plt.show()

# Atores com o melhor histórico de faturamento

mean_gross_by_ator = filtered_atores.groupby('Star1')['Gross'].mean().sort_values(ascending=False).head(20)
mean_gross_by_ator.plot(kind='barh', figsize=(10, 10), title
='Atores com o melhor histórico de faturamento')
plt.title('Atores com o melhor histórico de faturamento')
plt.ylabel('Atores')
plt.xlabel('Faturamento')
plt.show()


# Pré-processamento do texto

# Removendo stopwords


text = list(df['Overview'])


# nltk.download('stopwords')
# nltk.download('wordnet')


import re

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

corpus = []
for i in range(len(text)):
    r = re.sub('[^a-zA-Z]', ' ', text[i])
    r = r.lower()
    r = r.split()
    r = [word for word in r if word not in stopwords.words('english')]
    r = [lemmatizer.lemmatize(word) for word in r]
    r = ' '.join(r)
    corpus.append(r)

df['Overview'] = corpus

df.head()

# separando por palavras

words = []
for i in range(len(df)):
    words += df['Overview'][i].split()


# Plotando o gráfico
labels, values = zip(*Counter(words).most_common(10))
plt.pie(values, labels=labels, autopct='%1.1f%%')
plt.title('Palavras mais comuns nos resumos dos filmes')
plt.show()

# Palavras mais encontradas nos 10 melhores filmes IMDB

top_10_imdb = df.sort_values('IMDB_Rating', ascending=False).head(10)
words = []
for i in range(len(top_10_imdb)):
    words += top_10_imdb['Overview'][i].split()

labels, values = zip(*Counter(words).most_common(10))
plt.pie(values, labels=labels, autopct='%1.1f%%')
plt.title('Palavras mais comuns nos resumos dos 10 melhores filmes IMDB')
plt.show()

# Relação do Lucro com as Notas

# Relação entre lucro e notas IMDB
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='IMDB_Rating', y='Gross')
plt.title('Relação entre Lucro e Nota IMDB')
plt.xlabel('Nota IMDB')
plt.ylabel('Lucro')
plt.show()

# Relação entre lucro e notas Metascore
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Meta_score', y='Gross')
plt.title('Relação entre Lucro e Nota Metascore')
plt.xlabel('Nota Metascore')
plt.ylabel('Lucro')
plt.show()

