import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from data import load_pima
from preprocessing import preprocess
from models import train_models
from plots import plot_roc, plot_confusion, plot_feature_importance
from utils.save_results import save_all

def criar_diretorios():
    diretorios = [
        "outputs",
        "outputs/graficos",
        "outputs/modelos", 
        "outputs/resultados",
        "outputs/dados"
    ]
    
    for diretorio in diretorios:
        if not os.path.exists(diretorio):
            os.makedirs(diretorio)
            print(f"📁 Diretório criado: {diretorio}")
        else:
            print(f"✓ Diretório já existe: {diretorio}")

def main():
    print("=" * 60)
    print("PREDIÇÃO DE DIABETES - PIMA INDIANS DATASET")
    print("=" * 60)
    
    print("\n📂 CONFIGURANDO DIRETÓRIOS...")
    criar_diretorios()
    
    print("\n📊 CARREGANDO DADOS...")
    df = load_pima()
    print(f"✅ Dataset carregado. Dimensões: {df.shape}")
    print(f"   Colunas: {list(df.columns)}")
    
    print("\n🔄 PRÉ-PROCESSANDO DADOS...")
    X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler = preprocess(df)
    print(f"✅ Pré-processamento concluído.")
    print(f"   Treino: {X_train.shape[0]} amostras")
    print(f"   Teste: {X_test.shape[0]} amostras")
    
    print("\n🤖 TREINANDO MODELOS...")
    lr, m_lr, cm_lr, pred_lr, proba_lr, rf, m_rf, cm_rf, pred_rf, proba_rf = train_models(
        X_train_s, X_test_s, X_train, X_test, y_train, y_test
    )
    print("✅ Modelos treinados com sucesso!")
    print(f"   Regressão Logística - AUC: {m_lr.get('AUC', 'N/A'):.3f}")
    print(f"   Random Forest - AUC: {m_rf.get('AUC', 'N/A'):.3f}")
    
    print("\n🎨 GERANDO GRÁFICOS...")
    plot_roc(y_test, proba_lr, proba_rf, m_lr["AUC"], m_rf["AUC"], "outputs/graficos/roc.png")
    plot_confusion(cm_lr, cm_rf, "outputs/graficos/matrizes_confusao.png")
    plot_feature_importance(rf, X_train.columns.tolist(), "outputs/graficos/importancia_features.png")
    
    print("\n💾 SALVANDO RESULTADOS...")
    save_all(m_lr, m_rf, lr, rf, scaler, X_train.columns.tolist())
    
    print("\n" + "=" * 60)
    print("✅ EXECUÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 60)
    print("\n📁 ARQUIVOS GERADOS:")
    print("  outputs/graficos/roc.png")
    print("  outputs/graficos/matrizes_confusao.png")
    print("  outputs/graficos/importancia_features.png")
    print("  outputs/resultados/metricas_modelos.csv")
    print("  outputs/resultados/relatorio_detalhado.txt")
    print("  outputs/modelos/ (modelos salvos em .pkl)")
    print("\n📊 RESULTADOS PRINCIPAIS:")
    print(f"  Random Forest: AUC-ROC = {m_rf.get('AUC', 'N/A'):.3f}")
    print(f"  Regressão Logística: AUC-ROC = {m_lr.get('AUC', 'N/A'):.3f}")

if __name__ == "__main__":
    main()