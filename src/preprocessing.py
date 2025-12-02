import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(df):
    df = df.copy()
    cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    # Substituir zeros por NaN
    for c in cols:
        df[c] = df[c].replace(0, np.nan)
    
    # Calcular estatísticas antes de preencher
    media_glicose = df['Glucose'].mean()
    media_pressao = df['BloodPressure'].mean()
    media_imc = df['BMI'].mean()
    mediana_pele = df['SkinThickness'].median()
    mediana_insulina = df['Insulin'].median()
    
    # Preencher valores ausentes
    df['Glucose'] = df['Glucose'].fillna(media_glicose)
    df['BloodPressure'] = df['BloodPressure'].fillna(media_pressao)
    df['BMI'] = df['BMI'].fillna(media_imc)
    df['SkinThickness'] = df['SkinThickness'].fillna(mediana_pele)
    df['Insulin'] = df['Insulin'].fillna(mediana_insulina)
    
    # Separar features e target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Divisão estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Padronização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler