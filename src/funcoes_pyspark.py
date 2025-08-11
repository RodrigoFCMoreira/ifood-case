from pyspark.sql import DataFrame
from pyspark.sql import functions as f, Window

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

def filtra_mais_recente(book_df):
    """
    Mantém apenas o registro mais recente de cada (cliente_id, offer_id) não nulo.
    """
    w = Window.partitionBy("cliente_id", "offer_id").orderBy(f.col("time_since_test_start").desc())
    
    return (book_df
            .filter(f.col("offer_id").isNotNull())
            .withColumn("rn", f.row_number().over(w))
            .filter(f.col("rn") == 1)
            .drop("rn")
           )