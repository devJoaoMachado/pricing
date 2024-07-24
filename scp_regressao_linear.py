from scp_gerar_dados_treinamento import gerar_dados
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Gerar dados
df = gerar_dados(num_samples=1000)

# Exibir os primeiros registros para verificar
print(df.head())

# Preprocessamento
# Convertendo variáveis categóricas para numéricas usando get_dummies
df = pd.get_dummies(df, columns=['sexo', 'profissao'], drop_first=True)

# Definir variáveis independentes (X) e dependentes (y)
X = df[['idade', 'sexo_M', 'profissao_engenheiro civil', 'profissao_policial militar']]
y = df['preco']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo de Regressão Linear
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Fazer previsões
y_pred = modelo.predict(X_test)

# Calcular métricas de avaliação
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\nMean Squared Error: {mse}')
print(f'R² Score: {r2}')

# Coeficientes do modelo
print('\nCoeficientes:', modelo.coef_)
print('Intercepto:', modelo.intercept_)

# Comparar valores reais e previstos
plt.scatter(y_test, y_pred)
plt.xlabel('Valores Reais (x)')
plt.ylabel('Valores Previstos (y)')
plt.title('Valores Reais vs. Valores Previstos')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()