# %%
import pandas as pd

df = pd.read_parquet('../dados/dados_clones.parquet')
df
# %%
# Como podemos descobrir onde está o problema?
# Estatística descritiva

df.groupby(['Status '])[['Estatura(cm)', 'Massa(em kilos)']].mean()
#%%
df['Status_bool'] = df['Status '] == 'Apto'
df
# %%
# Fazendo as tabelas de frequência
# Verificando se a coluna 'Distância Ombro a ombro' tem algum valor impactante para análise
df.groupby(['Distância Ombro a ombro'])['Status_bool'].mean()

#%%
# Verificando se a coluna 'Tamanho do crânio' tem algum valor impactante para análise
df.groupby(['Tamanho do crânio'])['Status_bool'].mean()

#%%
# Verificando se a coluna 'Tamanho dos pés' tem algum valor impactante para análise
df.groupby(['Tamanho dos pés'])['Status_bool'].mean()

# %%
# Verificando se a coluna 'General Jedi encarregado' tem algum valor impactante para análise
df.groupby(['General Jedi encarregado'])['Status_bool'].mean()

# %%

features = ['Estatura(cm)', 
            'Massa(em kilos)',
            'Distância Ombro a ombro',
            'Tamanho do crânio',	
            'Tamanho dos pés']

cat_features = ['Distância Ombro a ombro',
            'Tamanho do crânio',	
            'Tamanho dos pés']

X = df[features]

#%%
# Transformação de Categorias para Numérico
from feature_engine import encoding

onehot = encoding.OneHotEncoder(variables=cat_features)

onehot.fit(X)
X = onehot.transform(X)
X

# %%

# Treinando nossa árvore
from sklearn import tree
arvore = tree.DecisionTreeClassifier(max_depth=3)
arvore.fit(X, df['Status '])

# %%

# Plotando a árvore para análise
import matplotlib.pyplot as plt
plt.figure(dpi=600)
tree.plot_tree(arvore,
               class_names=arvore.classes_,
               feature_names=X.columns,
               filled=True,
               )
# %%
