import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import math
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp


# --------------------------
# Integração e pré-processamento de dados
# --------------------------

# Carrega a base de dados "loan_data.csv"
data = pd.read_csv("archive/loan_data.csv")

# Escolhe 200 exemplos aleatórios apenas para testar código.
# Comentar se for rodar para valer:
data = data.sample(n=200, random_state=42)



# resumo de cada coluna
for col in data.columns:
    print(f"Resumo da coluna: {col}")
    if pd.api.types.is_numeric_dtype(data[col]):
        # Para colunas numéricas, exibe mínimo, máximo, média e desvio padrão
        resumo = data[col].agg(['min', 'max', 'mean', 'std'])
        print(resumo)
    else:
        # Para colunas categóricas, exibe o número de valores únicos e a lista dos mesmos
        num_unicos = data[col].nunique()
        unicos = data[col].unique()
        print(f"Quantidade de valores únicos: {num_unicos}")
        print(f"Valores únicos: {unicos}")
    print("-" * 50)


# Verifica a quantidade de dados ausentes em cada coluna
missing_data = data.isnull().sum()
print("Dados ausentes por coluna:")
print(missing_data)

# mostra o percentual de dados ausentes em cada coluna
percent_missing = (data.isnull().sum() / len(data)) * 100
print("\nPercentual de dados ausentes por coluna:")
print(percent_missing)



















# Verifica a distribuição da variável de previsão 'loan_status'
print("Contagem de cada classe:")
print(data['loan_status'].value_counts())

print("\nPercentual de cada classe:")
print(data['loan_status'].value_counts(normalize=True) * 100)

# plot para visualizar a distribuição
data['loan_status'].value_counts().plot(kind='bar')
plt.title("Distribuição da variável 'loan_status'")
plt.xlabel("Loan Status")
plt.ylabel("Frequência")
plt.show()



# Somente se for necessário rebalancear os dados
# Filtra os dados para cada classe do atributo 'loan_status'
dados_classe0 = data[data['loan_status'] == 0]
dados_classe1 = data[data['loan_status'] == 1]

# Calcula o número mínimo de amostras disponíveis entre as classes
min_amostras = min(len(dados_classe0), len(dados_classe1))
print(f"Disponíveis: Classe 0 = {len(dados_classe0)}, Classe 1 = {len(dados_classe1)}. Mínimo = {min_amostras}")

# Número de exemplos desejados para cada classe (0 e 1)
# deve ser igual ou menor que o mínimo do print anterior
numero_cada_caso = 1000

# Verifica se há amostras suficientes em cada classe
if len(dados_classe0) < numero_cada_caso or len(dados_classe1) < numero_cada_caso:
    raise ValueError("Não há amostras suficientes em uma ou ambas as classes para o número especificado.")

# Amostra aleatoriamente o número desejado de cada classe
amostra_classe0 = dados_classe0.sample(n=numero_cada_caso, random_state=42)
amostra_classe1 = dados_classe1.sample(n=numero_cada_caso, random_state=42)

# Concatena as amostras e embaralha o dataset resultante
data_balanceada = pd.concat([amostra_classe0, amostra_classe1]).sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset balanceado:")
print(data_balanceada['loan_status'].value_counts())











# Pairplot geral
# Demora para rodar
g = sns.pairplot(data, diag_kind='hist')  # diag_kind='hist' ou 'kde'
g.fig.set_size_inches(10, 10)
plt.show()






# Pairplot apenas com colunas de interesse
colunas_interesse = ['person_age', 'person_gender', 'person_education',
                     'person_income', 'person_emp_exp', 'loan_amnt',
                     'loan_int_rate', 'loan_percent_income',
                     'cb_person_cred_hist_length', 'credit_score']
# Seleciona apenas essas colunas do DataFrame
dados_interesse = data[colunas_interesse]
g = sns.pairplot(dados_interesse, diag_kind='hist')
g.fig.set_size_inches(10, 10)
plt.show()


















# Transformação de dados:

# 1. person_gender - Variável binária
# Mapeando 'female' para 0 e 'male' para 1
data['person_gender'] = data['person_gender'].map({'female': 0, 'male': 1})

# 2. person_education - Variável ordinal
education_order = ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']
data['person_education'] = pd.Categorical(data['person_education'],
                                            categories=education_order,
                                            ordered=True).codes

# 3. person_home_ownership - One-hot encoded
# data = pd.get_dummies(data, columns=['person_home_ownership'], drop_first=True)
data = pd.get_dummies(data, columns=['person_home_ownership'], drop_first=False)

# 4. loan_intent - One-hot encoded
data = pd.get_dummies(data, columns=['loan_intent'], drop_first=False)

# 5. previous_loan_defaults_on_file - Variável binária
# Mapeando 'No' para 0 e 'Yes' para 1
data['previous_loan_defaults_on_file'] = data['previous_loan_defaults_on_file'].map({'No': 0, 'Yes': 1})

# Move a coluna "loan_status" para o final
loan_status = data.pop("loan_status")
data["loan_status"] = loan_status

# Exibe as primeiras linhas para conferir as transformações
print(data.head())


















# deixar para normalizar dentro do pipeline
# cada modelo trbalha melhor com um tipo de normalização, normalizar tudo antes pode prejudicar o desempenho. Vale testar?
# Se normalizar aqui, a predição, ao final, tem que ser inserida normalizada, o que pode ser um problema para o usuário final.

# # Normalização do dataframe
# # ---------------------------
# features = data.iloc[:, :-1]  # Todas as colunas, exceto a última
# target = data.iloc[:, -1]     # A última coluna (atributo alvo)

# # 1. Utilizando StandardScaler
# # ---------------------------
# # Média 0 e variância 1 (supondo distribuição normal)
# # ---------------------------
# scaler_standard = StandardScaler()
# features_standard = pd.DataFrame(scaler_standard.fit_transform(features), columns=features.columns)
# # Reagrupa as features normalizadas com o atributo alvo
# data_standard = pd.concat([features_standard, target.reset_index(drop=True)], axis=1)
# print("Dados padronizados (StandardScaler):")
# print(data_standard.head())

# # 2. Utilizando MinMaxScaler
# # ---------------------------
# # Normalizar para valores entre 0 e 1
# # ---------------------------
# scaler_minmax = MinMaxScaler()
# features_minmax = pd.DataFrame(scaler_minmax.fit_transform(features), columns=features.columns)
# # Reagrupa as features normalizadas com o atributo alvo
# data_minmax = pd.concat([features_minmax, target.reset_index(drop=True)], axis=1)
# print("\nDados normalizados (MinMaxScaler):")
# print(data_minmax.head())





















# Classificação ---------------------------------------------------------------
# (onde o alvo são categorias, como 0 e 1)

import numpy as np
import pandas as pd

# Importa os classificadores e ferramentas para criação dos pipelines
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#from sklearn.neural_network import MLPClassifier  # Opcional

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer


# Define um seed para reprodutibilidade
seed = 42

# Define o scorer
scorer = make_scorer(accuracy_score)     # acurácia
# scorer = make_scorer(precision_score)  # precisão (quantos dos positivos previstos são realmente positivos)
# scorer = make_scorer(recall_score)     #  sensibilidade (recall) (quantos dos positivos reais foram identificados)
# scorer = make_scorer(f1_score)         # equilíbrio entre precisão e recall (média harmônica entre os dois)


# Estratégia de validação cruzada
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)  # Para avaliação final
gscv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)  # Para grid search





# # Separa as features (todas as colunas, exceto a última) e o target (última coluna)
# # Precisa escolher se quer usar data_minmax ou data_standard
# X = data_minmax.iloc[:, :-1]
# y = data_minmax.iloc[:, -1]


# Usa os dados "crus" (após as transformações categóricas e dummies)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# # Retira colunas que, porventura, tenham valores constantes
# data = data.loc[:, data.apply(pd.Series.nunique) > 1]


# Define um dicionário com os modelos (pipelines) e seus hiperparâmetros para busca em grade
# Define a variável de paralelização (n_jobs)

"""
Observações sobre a normalização nos modelos:

- kNN: Utiliza MinMaxScaler, que escala os dados para o intervalo [0, 1]. 
       Isso é útil, pois o kNN é sensível às diferenças de escala.

- tree e bigtree (Decision Trees): Não utilizam escalonamento.
       Modelos baseados em árvore não exigem normalização dos dados.

- nb (Naive Bayes): Não utiliza escalonamento, pois o modelo trabalha bem com os dados na escala original.

- svmlinear e svmrbf (SVM): Utilizam StandardScaler, que transforma os dados para média 0 e desvio padrão 1.
       Essa normalização geralmente melhora o desempenho dos SVM.
"""


jobs = -1  # Use -1 para usar todos os núcleos disponíveis, ou defina um número específico

algorithms = {
    'kNN': GridSearchCV(
        Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', MinMaxScaler(feature_range=(0, 1))),
            ('selector', VarianceThreshold()),
            ('knn', KNeighborsClassifier())
        ]),
        param_grid={
            'selector__threshold': [0, 0.01, 0.02, 0.03],
            'knn__n_neighbors': [1, 3, 5],
            'knn__p': [1, 2],
        },
        scoring=scorer,
        cv=gscv,
        n_jobs=jobs
    ),
    'tree': GridSearchCV(
        Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('tree', DecisionTreeClassifier(random_state=seed))
        ]),
        param_grid={
            'tree__max_depth': [5, 10, 20],
            'tree__criterion': ['entropy', 'gini'],
        },
        scoring=scorer,
        cv=gscv,
        n_jobs=jobs
    ),
    'bigtree': GridSearchCV(
        Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('tree', DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=seed))
        ]),
        param_grid={
            'tree__criterion': ['entropy', 'gini'],
        },
        scoring=scorer,
        cv=gscv,
        n_jobs=jobs
    ),
    'nb': GridSearchCV(
        Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('selector', SelectKBest()),
            ('nb', GaussianNB())
        ]),
        param_grid={
            'selector__k': [3, 5, 10],
        },
        scoring=scorer,
        cv=gscv,
        n_jobs=jobs
    ),
    'svmlinear': GridSearchCV(
        Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('pca', PCA()),
            ('svm', SVC(kernel='linear', random_state=seed))
        ]),
        param_grid={
            'pca__n_components': [2, 5, 10],
            'svm__C': [1.0, 2.0],
        },
        scoring=scorer,
        cv=gscv,
        n_jobs=jobs
    ),
    'svmrbf': GridSearchCV(
        Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('pca', PCA(random_state=seed)),
            ('svm', SVC(kernel='rbf', random_state=seed))
        ]),
        param_grid={
            'pca__n_components': [2, 5, 10],
            'svm__C': [1.0, 2.0],
            'svm__gamma': [0.1, 1.0, 2.0],
        },
        scoring=scorer,
        cv=gscv,
        n_jobs=jobs
    ),
}

# Avalia cada modelo utilizando validação cruzada e armazena os resultados (acurácia de cada divisão)
# Essa é a parte mais demorada, não consegui rodar com as 45000 linhas ainda para ver quanto tempo leva
result = {}
for alg, clf in algorithms.items():
    result[alg] = cross_val_score(clf, X, y, cv=cv)

# Converte os resultados para um DataFrame para visualização
result_df = pd.DataFrame.from_dict(result)
print(result_df)

# Formata os resultados mostrando média e desvio padrão de cada modelo
formatted_results = result_df.apply(lambda x: "{:.2f} ± {:.2f}".format(x.mean(), x.std()))
print("\nResultados (acurácia):")
print(formatted_results)

# Plota um boxplot para comparar a distribuição das acurácias dos diferentes modelos
plt.boxplot([scores for alg, scores in result_df.items()])
plt.xticks(np.arange(1, result_df.shape[1] + 1), result_df.columns)
plt.ylabel("Acurácia")
plt.title("Comparação de Desempenho dos Modelos")
plt.show()











# Comparar desempenhos --------------------------------------------------------

from scipy.stats import wilcoxon

# Exemplo: result_df['kNN'] contém os scores do modelo kNN e result_df['tree'] os do modelo Decision Tree.

# --- Comparar apenas um par por vez ---
# Comparar desempenho entre kNN e Decision Tree usando o teste de Wilcoxon:
stat, p_value = wilcoxon(result_df['kNN'], result_df['tree'])
print("Teste de Wilcoxon entre kNN e tree:")
print("Estatística = {:.3f}, p-value = {:.3f}".format(stat, p_value))
# Se p_value < 0.05, a diferença entre os modelos é estatisticamente significativa.


# Para comparações múltiplas entre todos os modelos, uma abordagem é utilizar o teste de Friedman
# seguido por um teste post-hoc, como o de Nemenyi, ou utilizar ferramentas como o Orange3
# Temos em 'resultf' seja o DataFrame com os resultados (ex.: acurácia) de cada modelo para cada fold.
# Cada coluna corresponde a um algoritmo e cada linha a um fold da validação cruzada.
# Exemplo fictício:
# result = pd.DataFrame({
#     'kNN': [0.80, 0.82, 0.79, ...],
#     'tree': [0.75, 0.77, 0.76, ...],
#     'bigtree': [0.78, 0.80, 0.79, ...],
#     'nb': [0.73, 0.74, 0.72, ...],
#     'svmlinear': [0.81, 0.83, 0.82, ...],
#     'svmrbf': [0.82, 0.84, 0.83, ...],
# })

# --- Teste de Friedman ---
# Para o teste de Friedman, passamos os resultados de cada algoritmo (colunas) como argumentos separados.
friedman_stat, friedman_p = friedmanchisquare(*[result_df[col] for col in result_df.columns])
print("Teste de Friedman:")
print("  Estatística = {:.3f}, p-value = {:.3f}".format(friedman_stat, friedman_p))

# Se o p-value for menor que o nível de significância (por exemplo, 0.05), há diferença estatisticamente significativa entre os modelos.

# --- Teste Post-hoc de Nemenyi ---
# Utilizamos a função posthoc_nemenyi_friedman do scikit_posthocs para obter uma matriz de p-valores para comparações par a par.
# Interpretação:
# Em cada célula (i,j) da matriz, temos o p-valor da comparação entre o algoritmo i e o algoritmo j.
# Valores menores que 0.05 indicam que os algoritmos diferem significativamente.
nemenyi_results = sp.posthoc_nemenyi_friedman(result_df)
print("\nMatriz de p-valores do Teste de Nemenyi:")
print(nemenyi_results)

# --- Critical Difference (CD) ---
# --- Cálculo dos Ranks Médios e Critical Difference (CD) ---
# Para construir um diagrama de diferença crítica, primeiro calculamos os ranks médios de cada algoritmo.
avg_ranks = result_df.rank(axis=1, method='average').mean().sort_values()
print("\nRanks médios de cada algoritmo:")
print(avg_ranks)

# O Critical Difference (CD) pode ser calculado pela fórmula:
#   CD = q_alpha * sqrt( k*(k+1) / (6*N) )
# Onde:
#   - k é o número de algoritmos,
#   - N é o número de folds (ou datasets) usados,
#   - q_alpha é um valor crítico da distribuição de Studentized Range.
#
# Para alpha=0.05 e, por exemplo, k=6, um valor aproximado é q_alpha ≈ 2.728.
# (Esse valor varia conforme k; consulte tabelas específicas para maiores precisões.)
k = result_df.shape[1]
N = result_df.shape[0]
q_alpha = 2.728  # valor aproximado para alpha=0.05 com ~6 algoritmos
cd = q_alpha * math.sqrt(k * (k + 1) / (6 * N))
print("\nCritical Difference (CD) = {:.3f}".format(cd))

# Ordena os algoritmos de acordo com os ranks médios (menor rank = melhor desempenho)
algorithms_sorted = avg_ranks.index.tolist()
ranks_sorted = avg_ranks.values

# --- Diagrama de Diferença Crítica (Critical Difference Diagram) ---
# Diagrama simplificado onde os algoritmos são posicionados em uma linha de acordo com seus ranks médios.
# A linha vermelha indicará a extensão do CD. Se a diferença entre os ranks médios de dois algoritmos for
# menor que o CD, eles não diferem significativamente.

# Cria a figura do diagrama
plt.figure(figsize=(10, 3))
y_rank = 1.0
# Plota os pontos (círculos pretos) no eixo x = rank, y = y_rank
plt.plot(ranks_sorted, [y_rank]*len(ranks_sorted), 'o', markersize=10, color='black')
# Posiciona o nome de cada algoritmo acima (ou abaixo) do ponto
for i, alg in enumerate(algorithms_sorted):
    plt.text(ranks_sorted[i], y_rank + 0.07, alg,
             rotation=45, ha='left', va='bottom')
plt.xlabel('Rank Médio')
plt.title('Diagrama de Critical Difference')
# Desenha a linha do CD em outra altura (por exemplo, y = 0.5)
y_cd = 0.5
x_start = min(ranks_sorted)
plt.hlines(y_cd, x_start, x_start + cd, colors='red', linewidth=3)
plt.text(x_start + cd/2, y_cd - 0.1, f'CD = {cd:.2f}', color='red', ha='center')
# Ajusta os limites do eixo x e y
plt.xlim(x_start - 0.5, max(ranks_sorted) + 0.5)
plt.ylim(0, y_rank + 0.5)
# Remove ticks do eixo y (para ficar só a linha horizontal)
plt.yticks([])
plt.tight_layout()
plt.show()


# --- ROC curve ---
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(10, 8))

for alg_name, grid in algorithms.items():
    # Ajusta o GridSearchCV, se ainda não tiver sido feito
    grid.fit(X, y)
    
    # Usa o melhor estimador do GridSearchCV
    best_clf = grid.best_estimator_
    
    # Tenta obter as probabilidades para a classe positiva
    if hasattr(best_clf, "predict_proba"):
        y_scores = best_clf.predict_proba(X)[:, 1]
    elif hasattr(best_clf, "decision_function"):
        y_scores = best_clf.decision_function(X)
    else:
        print(f"{alg_name} não suporta predição de probabilidades ou decision_function.")
        continue
    
    # Calcula os valores da curva ROC
    fpr, tpr, thresholds = roc_curve(y, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f"{alg_name} (AUC = {roc_auc:.2f})")

# Linha de referência (AUC = 0.5)
plt.plot([0, 1], [0, 1], "k--", label="AUC = 0.5")
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.title("Curvas ROC dos Modelos")
plt.legend(loc="lower right")
plt.show()














# --- Melhor Modelo (Deploying) ---
# Treinar com toda a base de dados para depois colocar em produção

# --- Deploying: Escolha o Modelo ---

# Exibe os modelos disponíveis
print("Modelos disponíveis:")
for key in algorithms.keys():
    print("-", key)

# Altere a variável abaixo para selecionar o modelo desejado:
model_choice = 'tree'

# Seleciona o modelo escolhido
classifier = algorithms[model_choice]

# Treina o modelo utilizando todo o conjunto de dados
classifier.fit(X, y)

# Exibe o melhor estimador encontrado pelo GridSearchCV
print("\nMelhor estimador encontrado:")
print(classifier.best_estimator_)

# Simula um novo dado para predição: utiliza a primeira amostra de X, por exemplo
# Pode rodar x_new com várias linhas também, sem problemas.
x_new = X.iloc[0:10, :]
print("\nExemplo de dados para predição:")
print(x_new)

# Realiza a predição para o novo dado
prediction = classifier.predict(x_new)
print("\nPredição para o exemplo:", prediction)











# Regressão  ------------------------------------------------------------------
# problemas em que o alvo é contínuo (por exemplo, prever preços, temperaturas, etc.)
# Não é o caso do nosso problema





























