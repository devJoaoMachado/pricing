import numpy as np
import pandas as pd

# Configurar o número de amostras
def gerar_dados(num_samples = 10000):

    # Configurar o gerador de números aleatórios
    np.random.seed(42)  # Para reprodutibilidade

    # Gerar dados para 'idade'
    idades = np.random.randint(18, 56, size=num_samples)

    # Gerar dados para 'sexo': M ou F
    sexos = np.random.choice(['M', 'F'], size=num_samples)

    # Gerar dados para profissões
    profissoes = np.random.choice(['analista de sistemas', 'engenheiro civil', 'policial militar'], size=num_samples)

    # Gerar dados para 'preço'
    # Preço é mais alto com maior idade e para homens
    precos = (100 + 2 * idades + 50 * (sexos == 'M') + 50  + 
              10 * (profissoes == 'analista de sistemas') +
              20 * (profissoes == 'engenheiro civil') +
              60 * (profissoes == 'policial militar') +
              np.random.normal(0, 10, size=num_samples))

    # Criar o DataFrame
    df = pd.DataFrame({
        'idade': idades,
        'sexo': sexos,
        'profissao' : profissoes,
        'preco': precos
    })

    return df