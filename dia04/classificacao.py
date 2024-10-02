#%%
import pandas as pd
# %%

df = pd.read_excel('../dados/dados_cerveja_nota.xlsx')
df
# %%

df['Aprovado'] = df['nota'] >= 5

features = ['cerveja']
target = 'Aprovado'

df


# %%

from sklearn import linear_model
reg = linear_model.LogisticRegression(penalty=None,
                                      fit_intercept=True)

# Aprendizagem do modelo
reg.fit(df[features], df[target])

# Predição do modelo
reg_predict = reg.predict(df[features])
reg_predict

# %%

from sklearn import metrics

# Acurácia
reg_acc = metrics.accuracy_score(df[target], reg_predict)
print('Acurácia Reg Log.:', reg_acc)

# Precisão
reg_precision = metrics.precision_score(df[target], reg_predict)
print('Precisão Reg Log:', reg_precision)

# Recall Score
reg_recall = metrics.recall_score(df[target], reg_predict)
print('Recall Reg Recall Score:', reg_recall)

# %%
# Matriz de confusão

reg_conf = metrics.confusion_matrix(df[target], reg_predict)
reg_conf = pd.DataFrame(reg_conf,
                        index = ['False', 'True'],
                        columns = ['False', 'True'])
reg_conf
# %%

# Usando Árvore de Descisão

from sklearn import tree
arvore = tree.DecisionTreeClassifier(max_depth=2)

# Aprendizagem do modelo
arvore.fit(df[features], df[target])

# Predição do modelo
arvore_predict = arvore.predict(df[features])
arvore_predict

# Acurácia
arvore_acc = metrics.accuracy_score(df[target], arvore_predict)
print('Acurárcia Árvore:', arvore_acc)

arvore_precision = metrics.precision_score(df[target], arvore_predict)
print('Precisão Árvore.:', arvore_precision)

# Recall Score
arvore_recall = metrics.recall_score(df[target], arvore_predict)
print('Árvore Recall Score:', arvore_recall)

arvore_conf = metrics.confusion_matrix(df[target], arvore_predict)
arvore_conf = pd.DataFrame(arvore_conf,
                        index = ['False', 'True'],
                        columns = ['False', 'True'])
arvore_conf

# %%
# Usando Naive Bayes

from sklearn import naive_bayes
nb = naive_bayes.GaussianNB()

# Aprendizagem do modelo
nb.fit(df[features], df[target])

# Predição do modelo
nb_predict = nb.predict(df[features])
nb_predict

# Acurácia
nb_acc = metrics.accuracy_score(df[target], nb_predict)
print('Acurárcia NB:', nb_acc)

nb_precision = metrics.precision_score(df[target], nb_predict)
print('Precisão Naive Bayes:', nb_precision)

# Recall Score
nb_recall = metrics.recall_score(df[target], nb_predict)
print('Naive Bayes Recall Score:', nb_recall)

nb_conf = metrics.confusion_matrix(df[target], nb_predict)
nb_conf = pd.DataFrame(arvore_conf,
                        index = ['False', 'True'],
                        columns = ['False', 'True'])
nb_conf

# %%
nb_predict
# %%
nb_proba = nb.predict_proba(df[features])[:,1]
nb_proba
# %%
df[features].iloc[0]
# %%

df['prob_nb'] = nb_proba
df.to_clipboard()
# %%

import matplotlib.pyplot as plt

roc_curve = metrics.roc_curve(df[target], nb_proba)
plt.plot(roc_curve[0], roc_curve[1])
plt.grid(True)
plt.plot([0, 1], [0, 1], '--')
plt.show()
# %%

roc_auc = metrics.roc_auc_score(df[target], nb_proba)
roc_auc
# %%
