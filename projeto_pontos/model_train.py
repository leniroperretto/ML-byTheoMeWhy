# %%
import pandas as pd

from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline

from sklearn import tree
from sklearn import linear_model
from sklearn import ensemble
from sklearn import naive_bayes


from feature_engine import imputation

import scikitplot as skplt

#%%

df = pd.read_csv('../dados/dados_pontos.csv', sep = ';')
df
# %%

features = df.columns.tolist()[3:-1]
target = 'flActive'

# %%

X_train, X_test, y_train, y_test = model_selection.train_test_split(df[features],
                                                                    df[target],
                                                                    test_size=0.2,
                                                                    random_state=42,
                                                                    stratify=df[target])

print('Tx. Resposta Treino:', y_train.mean())
print('Tx. Resposta Teste:', y_test.mean())

# %%

X_train.isna().sum().T

#%%

max_avgRecorrencia = X_train['avgRecorrencia'].max()


#%%
# Imputer

features_imput_0 = [
    'qtdeRecencia',
    'freqDias',
    'freqTransacoes',
    'qtdListaPresença',
    'qtdChatMessage',
    'qtdTrocaPontos',
    'qtdResgatarPonei',
    'qtdPresençaStreak',
    'pctListaPresença',
    'pctChatMessage',
    'pctTrocaPontos',
    'pctResgatarPonei',
    'pctPresençaStreak',
    'qtdePontosGanhos',
    'qtdePontosGastos',
    'qtdePontosSaldo',
]

imputacao_0 = imputation.ArbitraryNumberImputer(variables=features_imput_0,
                                                arbitrary_number=0)

imputacao_max = imputation.ArbitraryNumberImputer(variables=['avgRecorrencia'],
                                                  arbitrary_number=max_avgRecorrencia)

model = ensemble.RandomForestClassifier(random_state=42)

params = {
    "n_estimators": [100,150,250,500],
    "min_samples_leaf": [10,20,30,50,100],
}

grid = model_selection.GridSearchCV(model,
                                    param_grid=params,
                                    n_jobs=-1,
                                    scoring='roc_auc')

meu_pipeline = pipeline.Pipeline([
    ('imput_0', imputacao_0),
    ('imput_max', imputacao_max),
    ('model', grid),    
    ])

meu_pipeline.fit(X_train, y_train)

#%%

pd.DataFrame(grid.cv_results_)

#%%
y_test_predict = meu_pipeline.predict(X_test)
y_test_proba = meu_pipeline.predict_proba(X_test)

y_train_predict = meu_pipeline.predict(X_train)
y_train_proba = meu_pipeline.predict_proba(X_train)[:,1]

# %%

acc_train = metrics.accuracy_score(y_train, y_train_predict)
acc_test = metrics.accuracy_score(y_test, y_test_predict)
print('Acurácia base train:', acc_train)
print('Acurácia base teste:', acc_test)

auc_train = metrics.roc_auc_score(y_train, y_train_proba)
auc_test = metrics.roc_auc_score(y_test, y_test_proba[:,1])
print('\nAUC base train:', auc_train)
print('AUC base teste:', auc_test)


# Acurácia base train: 0.8109619686800895
# Acurácia base teste: 0.8008948545861297
# AUC base train: 0.8531284015204619
# AUC base teste: 0.8380512447094162

# %%

f_importance = meu_pipeline[-1].best_estimator_.feature_importances_
pd.Series(f_importance, index=features).sort_values(ascending=False)
# %%

import matplotlib.pyplot as plt
plt.figure(dpi=600)
skplt.metrics.plot_roc_curve(y_test, y_test_proba)
plt.show()

# %%

usuarios_test = pd.DataFrame(
    {'verdadeiro': y_test,
     'proba': y_test_proba[:,1]
    }
)

usuarios_test.sort_values('proba', ascending=False).head(20)
usuarios_test['sum_verdadeiro'] = usuarios_test['verdadeiro'].cumsum()
usuarios_test['tx captura'] = usuarios_test['sum_verdadeiro'] / usuarios_test['sum_verdadeiro'].sum()
usuarios_test
# %%

# LIFT
skplt.metrics.plot_lift_curve(y_test, y_test_proba)

# %%
