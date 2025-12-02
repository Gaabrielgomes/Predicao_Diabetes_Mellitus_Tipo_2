# Preditor de Diabetes - Pima Indians Dataset

## Descrição
Sistema de predição de diabetes mellitus tipo 2 utilizando algoritmos de Machine Learning (Regressão Logística e Random Forest) no dataset Pima Indians Diabetes.

## Objetivo
Comparar o desempenho de dois modelos clássicos de classificação para identificar pacientes com risco de diabetes.

## Tecnologias
- Python 3.10+
- Scikit-learn (modelos e métricas)
- Pandas (manipulação de dados)
- Matplotlib/Seaborn (visualização)
- Joblib (serialização de modelos)

## Dataset

**Download direto do dataset:**  
[🔗 pima-indians-diabetes.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv)

**Características:**
- 768 amostras
- 8 features clínicas
- 1 target binário (0 = não diabético, 1 = diabético)
- Formato CSV sem cabeçalho

**Exemplo de uso:**
```python
import pandas as pd
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
df = pd.read_csv(url, names=['Pregnancies','Glucose','BloodPressure','SkinThickness',
                             'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome'])

## Estrutura do Projeto
```
preditor_diabetes/
├── src/
│   ├── main.py              # Script principal
│   ├── data.py              # Carregamento de dados
│   ├── preprocessing.py     # Pré-processamento
│   ├── models.py           # Treinamento de modelos
│   ├── plots.py            # Geração de gráficos
│   └── save_results.py     # Salvamento de resultados
├── outputs/
│   ├── graficos/           # Imagens dos gráficos
│   ├── modelos/            # Modelos serializados (.pkl)
│   └── resultados/         # Métricas e relatórios
└── requirements.txt        # Dependências
```

## Instalação e Execução

### 1. Clonar e configurar
```bash
git clone <repositorio>
cd preditor_diabetes
```

### 2. Instalar dependências
```bash
pip install -r requirements.txt
```

### 3. Executar o projeto
```bash
cd src
python main.py
```

## Resultados Gerados
O sistema produz automaticamente:
- **Gráficos**: Curva ROC, Matrizes de Confusão, Importância das Features
- **Métricas**: Acurácia, Precisão, Recall, F1-Score, AUC-ROC
- **Modelos**: Arquivos `.pkl` para deploy
- **Relatórios**: Arquivos CSV e TXT com resultados detalhados

## Configuração dos Modelos
- **Regressão Logística**: `max_iter=1000`, `class_weight='balanced'`
- **Random Forest**: `n_estimators=100`, `max_depth=10`

## Métricas de Avaliação
- Acurácia
- Precisão
- Recall (Sensibilidade)
- F1-Score
- AUC-ROC

## Saídas Esperadas
Após execução, serão criados na pasta `outputs/`:
- `graficos/roc.png` - Comparação de curvas ROC
- `graficos/matrizes_confusao.png` - Matrizes de confusão
- `graficos/importancia_features.png` - Importância das features
- `modelos/` - Modelos treinados em formato .pkl
- `resultados/` - Arquivos CSV e TXT com métricas

## Pré-requisitos
- Python 3.10 ou superior
- 4GB RAM mínimo
- 500MB espaço em disco

## Fluxo de Processamento
1. Carregamento do dataset
2. Pré-processamento (imputação, padronização)
3. Divisão treino/teste (80/20)
4. Treinamento dos modelos
5. Avaliação de desempenho
6. Geração de visualizações
7. Salvamento de resultados

## Autores
- Eduardo Jorge
- Gabriel Calheiros  
- Julia Bertonha

## Referências
- UCI Machine Learning Repository
- Scikit-learn Documentation
- Pima Indians Diabetes Dataset
