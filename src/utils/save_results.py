import pandas as pd
import joblib
import os

def save_all(metrics_lr, metrics_rf, lr, rf, scaler, feature_names, path_base="outputs"):
    # Criar diretórios
    for subdir in ["modelos", "resultados", "graficos"]:
        diretorio = os.path.join(path_base, subdir)
        if not os.path.exists(diretorio):
            os.makedirs(diretorio)
    
    # 1. Salvar métricas em CSV
    df_metrics = pd.DataFrame([metrics_lr, metrics_rf], index=["Regressão_Logística", "Random_Forest"])
    caminho_metrics = os.path.join(path_base, "resultados", "metricas_modelos.csv")
    df_metrics.to_csv(caminho_metrics, encoding='utf-8')
    print(f"✅ Métricas salvas em: {caminho_metrics}")
    
    # 2. Salvar modelos
    caminho_lr = os.path.join(path_base, "modelos", "modelo_regressao_logistica.pkl")
    caminho_rf = os.path.join(path_base, "modelos", "modelo_random_forest.pkl")
    caminho_scaler = os.path.join(path_base, "modelos", "scaler.pkl")
    
    joblib.dump(lr, caminho_lr)
    joblib.dump(rf, caminho_rf)
    joblib.dump(scaler, caminho_scaler)
    print(f"✅ Modelos salvos em: {caminho_lr}, {caminho_rf}, {caminho_scaler}")
    
    # 3. Salvar relatório detalhado
    caminho_relatorio = os.path.join(path_base, "resultados", "relatorio_detalhado.txt")
    with open(caminho_relatorio, "w", encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("RELATÓRIO DE RESULTADOS - PREDIÇÃO DE DIABETES\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("RESULTADOS DOS MODELOS:\n")
        f.write("-" * 40 + "\n")
        f.write(df_metrics.to_string())
        
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("INFORMAÇÕES ADICIONAIS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total de amostras: {len(lr.classes_) if hasattr(lr, 'classes_') else 'N/A'}\n")
        f.write(f"Classes: {lr.classes_ if hasattr(lr, 'classes_') else 'N/A'}\n")
        f.write(f"Número de features: {len(feature_names)}\n")
        f.write(f"Features utilizadas: {', '.join(feature_names)}\n")
    
    print(f"✅ Relatório salvo em: {caminho_relatorio}")