# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd

# 1. Ruta al directorio de logs
log_dir = os.path.join('experiments', 'logs')

# 2. Encontrar solo los archivos de test (terminan en _tst.csv)
test_files = glob.glob(os.path.join(log_dir, '*_tst.csv'))

all_results = []

print(f"Encontrados {len(test_files)} archivos de testing. Compilando...")

for file in test_files:
    try:
        # --- A. Leer las Métricas de Test ---
        # Como se guardó como pd.Series, leemos usando la primera columna como índice
        df = pd.read_csv(file, index_col=0)
        
        # Convertir la serie/dataframe a un diccionario estándar
        test_metrics = df.squeeze("columns").to_dict()
        
        # --- B. Extraer Metadatos del Nombre del Archivo ---
        filename = os.path.basename(file)
        # Quitamos la extensión '_tst.csv' y dividimos por '-'
        name_parts = filename.replace('_tst.csv', '').split('-')
        
        if len(name_parts) >= 7:
            metadata = {
                'WDS': name_parts[0],
                'Deployment': name_parts[1],
                'Ratio': float(name_parts[2]),
                'Adjacency': name_parts[3],
                'GNN_Model': name_parts[4],
                'Tag': name_parts[5],
                'Run_ID': int(name_parts[6])
            }
        else:
            # Fallback en caso de que algún archivo no tenga el formato esperado
            metadata = {'Filename': filename, 'WDS': 'unknown_wds'}
            
        # --- C. Combinar y Almacenar ---
        combined_data = {**metadata, **test_metrics}
        all_results.append(combined_data)
        
    except Exception as e:
        print(f" Error procesando {file}: {e}")

# 3. Crear, Separar y Guardar DataFrames
if all_results:
    master_df = pd.DataFrame(all_results)
    
    # Ordenar los datos para que queden limpios
    if 'Ratio' in master_df.columns and 'GNN_Model' in master_df.columns:
        master_df = master_df.sort_values(by=['Ratio', 'GNN_Model', 'Run_ID'])
        
    # --- LÓGICA DE SEPARACIÓN POR RED (WDS) ---
    unique_wds = master_df['WDS'].unique()
    print("\n--- Guardando Archivos Individuales por Red (Testing) ---")
    
    for wds_name in unique_wds:
        # Filtrar el dataframe maestro solo para esta ciudad
        wds_df = master_df[master_df['WDS'] == wds_name]
        
        # Crear un nombre de archivo específico (ej. "anytown_Test_Summary.csv")
        output_file = f"{wds_name}_Test_Summary.csv"
        
        # Guardar el CSV
        wds_df.to_csv(output_file, index=False)
        print(f" Guardadas {len(wds_df)} ejecuciones en -> {output_file}")
        
else:
    print("\n No se encontraron archivos de testing (_tst.csv) en experiments/logs/.")