from pyspark.sql import DataFrame

def analisar_dataframe(df: DataFrame, id_col: str):
    # Contar linhas
    num_linhas = df.count()
    
    # Contar colunas
    num_colunas = len(df.columns)
    
    # Contar IDs distintos
    distinct_ids = df.select(id_col).distinct().count()
    
    # Prints claros
    print("==== Análise do DataFrame ====")
    print(f"Quantidade de linhas: {num_linhas}")
    print(f"Quantidade de colunas: {num_colunas}")
    print(f"Quantidade de {id_col} distintos: {distinct_ids}")
    print("==============================")
    
    # Retorna para uso posterior
    return num_linhas, num_colunas, distinct_ids