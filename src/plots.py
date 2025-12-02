import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
import os

def plot_roc(y_test, proba_lr, proba_rf, auc_lr, auc_rf, path):
    # Criar diretório se não existir
    diretorio = os.path.dirname(path)
    if diretorio and not os.path.exists(diretorio):
        os.makedirs(diretorio)
    
    # Calcular curvas ROC
    fpr_lr, tpr_lr, _ = roc_curve(y_test, proba_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, proba_rf)

    # Criar gráfico
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_lr, tpr_lr, label=f"Regressão Logística (AUC={auc_lr:.3f})", linewidth=2, color='blue')
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={auc_rf:.3f})", linewidth=2, color='green')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Classificador Aleatório')
    
    # Configurar gráfico
    plt.xlabel('Taxa de Falsos Positivos', fontsize=12)
    plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
    plt.title('Curva ROC - Comparação de Modelos', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Salvar figura
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()  # Fechar figura para liberar memória
    print(f"✅ Gráfico ROC salvo em: {path}")

def plot_confusion(cm_lr, cm_rf, path):
    # Criar diretório se não existir
    diretorio = os.path.dirname(path)
    if diretorio and not os.path.exists(diretorio):
        os.makedirs(diretorio)
    
    # Criar gráfico
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Matriz para Regressão Logística
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Não Diabético', 'Diabético'],
                yticklabels=['Não Diabético', 'Diabético'],
                ax=axes[0], cbar_kws={'shrink': 0.8})
    axes[0].set_title('Regressão Logística', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Classe Predita', fontsize=10)
    axes[0].set_ylabel('Classe Real', fontsize=10)
    
    # Matriz para Random Forest
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Não Diabético', 'Diabético'],
                yticklabels=['Não Diabético', 'Diabético'],
                ax=axes[1], cbar_kws={'shrink': 0.8})
    axes[1].set_title('Random Forest', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Classe Predita', fontsize=10)
    axes[1].set_ylabel('Classe Real', fontsize=10)
    
    plt.suptitle('Matrizes de Confusão', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Salvar figura
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()  # Fechar figura para liberar memória
    print(f"✅ Matrizes de confusão salvas em: {path}")

def plot_feature_importance(modelo_rf, feature_names, path):
    # Criar diretório se não existir
    diretorio = os.path.dirname(path)
    if diretorio and not os.path.exists(diretorio):
        os.makedirs(diretorio)
    
    # Extrair importâncias
    importancias = pd.DataFrame({
        'Feature': feature_names,
        'Importância': modelo_rf.feature_importances_
    }).sort_values('Importância', ascending=True)
    
    # Criar gráfico
    plt.figure(figsize=(10, 6))
    bars = plt.barh(importancias['Feature'], importancias['Importância'], 
                    color=plt.cm.viridis(np.linspace(0, 1, len(importancias))))
    
    # Adicionar valores nas barras
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='left', va='center', fontsize=9)
    
    plt.xlabel('Importância Relativa', fontsize=12)
    plt.title('Importância das Features - Random Forest', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    # Salvar figura
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Importância das features salva em: {path}")