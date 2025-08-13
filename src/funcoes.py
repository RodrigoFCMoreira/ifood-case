import math
from sklearn.metrics import precision_recall_curve, auc
#from pycaret.clustering import setup as clu_setup
#from pycaret.regression import setup as reg_setup
#from pycaret.classification import setup as cls_setup, create_model, tune_model
from sklearn.metrics import roc_curve, auc
from typing import Dict
from typing import Tuple, List
#from pycaret.classification import setup, create_model
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import seaborn as sns
from IPython.display import display
from typing import Any
#from pycaret.classification import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from typing import Dict, Tuple
from typing import List, Literal
#from pycaret.classification import ClassificationExperiment
from collections import Counter
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
#from optbinning import OptimalBinning
#from sklearn.cluster import KMeans
#from pycaret.classification import load_model, predict_model
#from scipy.spatial.distance import pdist, squareform


def perfil_base(base_modelo: pd.DataFrame, id_col: str, target_col: str, safra_col: str) -> dict:
    """
    Calcula métricas básicas do perfil da base de dados.

    Parâmetros:
    - base_modelo (pd.DataFrame): DataFrame contendo os dados a serem analisados.
    - id_col (str): Nome da coluna que representa o identificador único (ID).
    - target_col (str): Nome da coluna que representa a variável alvo (Y).
    - safra_col (str): Nome da coluna que representa a safra.

    Retorna:
    - dict: Dicionário contendo as seguintes métricas:
        - shape: Tupla com a quantidade de linhas e colunas.
        - tipos_variaveis: Contagem dos tipos das variáveis.
        - ids_unicos: Quantidade de IDs únicos.
        - bad_rate: Taxa de maus (bad rate) da base.
        - volumetria_safras: Quantidade de registros por safra.
    """

    perfil = {}

    # 1. Verificando a volumetria de linhas e colunas
    perfil['shape'] = f"Essa base possui {base_modelo.shape[0]} linhas e {base_modelo.shape[1]} colunas"

    # 2. Verificando a tipagem das variáveis
    perfil['tipos_variaveis'] = base_modelo.dtypes.value_counts().to_dict()

    # 3. Verificando a quantidade de IDs únicos (possíveis duplicatas)
    perfil['ids_unicos'] = base_modelo[id_col].nunique()

    # 4. Verificando a taxa de maus (bad rate)
    if target_col in base_modelo.columns:
        total_bons_maus = base_modelo[target_col].value_counts()
        bad_rate = total_bons_maus / perfil['ids_unicos']
        perfil['bad_rate'] = f"Bons: {total_bons_maus[0]}({round(bad_rate[0] * 100, 1)} %), Maus: {total_bons_maus[1]} ({round(bad_rate[1] * 100, 1)}%)"
    else:
        perfil['bad_rate'] = "Coluna alvo não encontrada."

    # 5. Verificando a quantidade de safras e suas volumetrias
    if safra_col in base_modelo.columns:
        perfil['volumetria_safras'] = dict(
            sorted(base_modelo[safra_col].value_counts().to_dict().items()))
    else:
        perfil['volumetria_safras'] = "Coluna safra não encontrada."

    print("Calcula métricas básicas do perfil da base de dados.")
    print(f"Shape da base: {perfil['shape']}")
    print(f"Tipos de variáveis: {perfil['tipos_variaveis']}")
    print(f"IDs únicos: {perfil['ids_unicos']}")
    print(f"Taxa de conversão: {perfil['bad_rate']}")
    print(f"Volumetria das safras: {perfil['volumetria_safras']}")
    print("\n")

    return perfil


def plot_safra_bad_rate(df: pd.DataFrame, safra_col: str = "safra", inadimplente_col: str = "y",
                        bad_rate_min: Optional[float] = None, bad_rate_max: Optional[float] = None) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    from typing import Optional

    # Garantir que safra seja numérica para ordenar corretamente
    df[safra_col] = pd.to_numeric(df[safra_col], errors="coerce")

    # Agrupar e ordenar
    safra_stats = (
        df.groupby(safra_col)
        .agg(
            contagem=(inadimplente_col, "count"),
            total_maus=(inadimplente_col, "sum")
        )
        .reset_index()
        .sort_values(safra_col)  # <-- Ordenação numérica
    )

    safra_stats["total_bons"] = safra_stats["contagem"] - safra_stats["total_maus"]
    safra_stats["badrate"] = safra_stats["total_maus"] / safra_stats["contagem"]

    # Criar figura e eixos
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Eixo principal (barras)
    ax1.bar(safra_stats[safra_col].astype(str), safra_stats["contagem"],
            color="blue", alpha=0.6, label="Total de Contratos")
    ax1.set_xlabel("Safra")
    ax1.set_ylabel("Total de IDs", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_xticklabels(safra_stats[safra_col].astype(str), rotation=45)

    # Eixo secundário (linha)
    ax2 = ax1.twinx()
    ax2.plot(safra_stats[safra_col].astype(str), safra_stats["badrate"],
             color="red", marker="o", linestyle="-", linewidth=2, label="Bad Rate")
    ax2.set_ylabel("Bad Rate (%)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    if bad_rate_min is not None and bad_rate_max is not None:
        ax2.set_ylim(bad_rate_min, bad_rate_max)

    plt.title("Total de IDs por Safra e Bad Rate")
    fig.tight_layout()
    plt.show()

    return safra_stats


def dividir_base_safra(df: pd.DataFrame, safra_corte: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide um dataframe em base de treino e teste OOT com base em uma safra de corte.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo a coluna 'safra'.
    safra_corte (int): Valor da safra que define o corte entre treino e teste OOT.

    Retorna:
    tuple[pd.DataFrame, pd.DataFrame]: 
        - treino (pd.DataFrame): Base de treino contendo safras anteriores ao safra_corte.
        - teste_oot (pd.DataFrame): Base de teste OOT contendo safras a partir do safra_corte.
    """

    # Garantindo que a coluna 'safra' seja do tipo inteiro
    df['safra'] = df['safra'].astype(int)

    # Separando a base de treino (safras menores que a safra de corte)
    treino: pd.DataFrame = df[df['safra'] < safra_corte]

    # Separando a base de teste OOT (safras maiores ou iguais à safra de corte)
    teste_oot: pd.DataFrame = df[df['safra'] >= safra_corte]

    # Criando uma tabela de volumetria para melhor visualização da divisão
    volumetria: pd.DataFrame = pd.DataFrame({
        'Conjunto': ['Treino', 'Teste OOT'],
        'Registros': [treino.shape[0], teste_oot.shape[0]]
    })

    # Exibindo os resultados
    print("\n### Volumetria da Base ###")
    print(volumetria)

    print("\n### Safras na Base de Treino ###")
    print(treino['safra'].value_counts().sort_index())

    print("\n### Safras na Base de Teste OOT ###")
    print(teste_oot['safra'].value_counts().sort_index())

    return treino, teste_oot


def remover_missings(df: pd.DataFrame, perc_miss: int = 20) -> pd.DataFrame:
    """
    Remove colunas que possuem um percentual de valores ausentes (missings) maior ou igual ao valor definido em perc_miss.

    Parâmetros:
    - df (pd.DataFrame): DataFrame de entrada.
    - perc_miss (int, opcional): Percentual máximo de valores ausentes permitido em uma coluna. 
      Colunas com valores ausentes acima desse percentual serão removidas. Padrão é 20.

    Retorna:
    - pd.DataFrame: DataFrame sem as colunas que ultrapassam o limite de valores ausentes.
    """
    qt_rows = df.shape[0]

    # Calcula a porcentagem de valores ausentes por coluna
    pct_missing = df.isnull().sum() / qt_rows * 100

    # Filtra as colunas que devem ser removidas
    colunas_removidas = pct_missing[pct_missing >= perc_miss].index.tolist()

    # Exibe a lista de colunas removidas
    if colunas_removidas:
        print(
            f"Colunas removidas({len(colunas_removidas)}): {colunas_removidas}")
    else:
        print("Nenhuma coluna removida.")

    # Retorna o DataFrame filtrado
    return df.drop(columns=colunas_removidas)


def escolher_estrategia_imputacao(df: pd.DataFrame) -> dict:
    """
    Função que determina a estratégia de imputação de valores ausentes para cada coluna de um DataFrame,
    com base no tipo da variável, presença de outliers e porcentagem de valores ausentes.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados a serem analisados.

    Retorna:
    dict: Um dicionário onde as chaves são os nomes das colunas e os valores são as estratégias de imputação.
    """
    estrategias = {}

    for coluna in df.columns:
        # Porcentagem de valores ausentes
        missing_pct = df[coluna].isna().mean()
        dtype = df[coluna].dtype  # Tipo de dado da coluna

        if dtype == 'object':  # Variável categórica
            estrategia = 'Moda'

        else:  # Variável numérica
            valores = df[coluna].dropna()

            # Identificando outliers usando IQR
            Q1, Q3 = np.percentile(valores, [25, 75])
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            tem_outliers = ((valores < limite_inferior) |
                            (valores > limite_superior)).any()

            # Definição da estratégia baseada em missing_pct e outliers
            if tem_outliers and 0.05 <= missing_pct <= 0.20:
                estrategia = 'median'
            elif not tem_outliers and missing_pct < 0.05:
                estrategia = 'mean'
            else:
                estrategia = 'median'  # Estratégia segura para outros casos

        estrategias[coluna] = estrategia

    return estrategias


def aplicar_imputacao_treino(df: pd.DataFrame, regra_imputacao: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, float], Dict[str, float]]:
    """
    Aplica imputação de valores ausentes em um DataFrame com base em uma regra especificada.

    Parâmetros:
    - df (pd.DataFrame): DataFrame contendo os dados a serem processados.
    - regra_imputacao (Dict[str, str]): Dicionário onde as chaves são os nomes das colunas do DataFrame 
      e os valores são as regras de imputação ('median' para mediana ou 'mean' para média).

    Retorna:
    - Tuple contendo:
        1. O DataFrame com os valores ausentes imputados.
        2. O dicionário de regras de imputação utilizado.
        3. O dicionário com os valores de mediana calculados por coluna.
        4. O dicionário com os valores de média calculados por coluna.

    Obs: Se a coluna informada na regra de imputação não existir no DataFrame, uma mensagem será exibida.

    """

    # Criando uma cópia do DataFrame para evitar modificações no original
    df_copy = df.copy()

    # Calcula os valores de mediana e média de cada coluna
    dict_mediana: Dict[str, float] = df_copy.median().to_dict()
    dict_media: Dict[str, float] = df_copy.mean().to_dict()

    # Itera sobre as colunas do DataFrame e aplica a imputação conforme a regra especificada
    for col in df_copy.columns:
        if col in regra_imputacao:
            if regra_imputacao[col] == 'median':
                df_copy[col] = df_copy[col].fillna(dict_mediana[col])
            elif regra_imputacao[col] == 'mean':
                df_copy[col] = df_copy[col].fillna(dict_media[col])
        else:
            print(
                f"A regra de imputação para a coluna '{col}' não foi especificada.")

    return df_copy, regra_imputacao, dict_mediana, dict_media


def aplicar_imputacao_teste(df: pd.DataFrame,
                            regra_imputacao: Dict[str, str],
                            dict_mediana: Dict[str, float],
                            dict_media: Dict[str, float]) -> pd.DataFrame:
    """
    Aplica imputação de valores ausentes em um DataFrame com base em regras especificadas.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo os dados com valores ausentes a serem imputados.

    regra_imputacao : Dict[str, str]
        Dicionário onde as chaves são os nomes das colunas e os valores indicam a regra de imputação:
        - 'median' para imputação com a mediana.
        - 'mean' para imputação com a média.

    dict_mediana : Dict[str, float]
        Dicionário contendo as medianas das colunas a serem imputadas.

    dict_media : Dict[str, float]
        Dicionário contendo as médias das colunas a serem imputadas.

    Retorno:
    --------
    pd.DataFrame
        DataFrame com os valores ausentes imputados conforme as regras definidas.
    """

    # Itera sobre as colunas do DataFrame e aplica a imputação conforme a regra especificada
    for col in df.columns:
        if col in regra_imputacao:
            if regra_imputacao[col] == 'median':
                df[col] = df[col].fillna(dict_mediana.get(col, df[col]))
            elif regra_imputacao[col] == 'mean':
                df[col] = df[col].fillna(dict_media.get(col, df[col]))
        else:
            print(
                f"A regra de imputação para a coluna '{col}' não foi especificada.")

    return df


def selecao_variaveis(
    data: pd.DataFrame,
    target: str,
    methods: List[Literal['classic', 'univariate', 'sequential']],
    selection_rule: Literal['intersection', 'union', 'voting'] = 'intersection'
) -> List[str]:
    """
    Realiza a seleção de variáveis no PyCaret usando diferentes métodos e regras de combinação.

    Parâmetros:
    - data (pd.DataFrame): Dataset contendo as variáveis preditoras e a variável alvo.
    - target (str): Nome da variável alvo.
    - methods (List[str]): Lista com os métodos de seleção a serem aplicados. Opções:
        - 'classic' (RFE - Recursive Feature Elimination)
        - 'univariate' (Testes estatísticos ANOVA/qui-quadrado)
        - 'sequential' (Sequential Feature Selection - SFS)
    - selection_rule (str): Método de combinação das variáveis selecionadas. Opções:
        - 'intersection': Mantém apenas as variáveis escolhidas por todos os métodos.
        - 'union': Mantém todas as variáveis selecionadas por pelo menos um método.
        - 'voting': Mantém variáveis selecionadas por pelo menos 2 dos métodos escolhidos.

    Retorno:
    - List[str]: Lista final de variáveis selecionadas.
    """

    """
    Observações relevantes sobre o uso:
    Bases Pequenas/Médias (até 1000 variáveis) → classic ou sequential
    Se precisar de um modelo bem ajustado → sequential.
    Se quiser um método robusto baseado no impacto real das variáveis → classic.

    Bases Grandes (acima de 5000 variáveis) → univariate
    Se a base é muito grande, o univariate é mais rápido e ajuda a filtrar variáveis antes de rodar modelos mais pesados.
    Depois, pode usar classic ou sequential só nas melhores variáveis.
    
    """

    # Verifica se os métodos fornecidos são válidos
    valid_methods = {'classic', 'univariate', 'sequential'}
    if not set(methods).issubset(valid_methods):
        raise ValueError(
            f"Os métodos devem estar entre {valid_methods}, mas recebeu {methods}")

    selected_features_sets = []

    for method in methods:
        exp = ClassificationExperiment()  # Inicializa o experimento
        exp.setup(data, target=target, feature_selection=True,
                  feature_selection_method=method, verbose=False)

        # Pegamos as features selecionadas via get_config
        selected_features = exp.get_config("X_train").columns.to_list()
        selected_features_sets.append(set(selected_features))

    # Combinação das seleções
    if selection_rule == 'intersection':
        variaveis_selecionadas = list(
            set.intersection(*selected_features_sets))
    elif selection_rule == 'union':
        variaveis_selecionadas = list(set.union(*selected_features_sets))
    elif selection_rule == 'voting':
        feature_counts = Counter(
            [feat for features in selected_features_sets for feat in features])
        variaveis_selecionadas = [feat for feat,
                                  count in feature_counts.items() if count >= 2]
    else:
        raise ValueError(
            "selection_rule deve ser 'intersection', 'union' ou 'voting'")

    return variaveis_selecionadas


def resumo_estatistico(df: pd.DataFrame) -> None:
    """
    Exibe um resumo estatístico das variáveis numéricas e categóricas do DataFrame.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados.
    """
    df_numeric = df.select_dtypes(include=[np.number])
    df_categoric = df.select_dtypes(include=["O"])

    if not df_numeric.empty:
        print("📌 Resumo Estatístico das Variáveis Numéricas:")
        display(df_numeric.describe())

    if not df_categoric.empty:
        print("\n📌 Resumo Estatístico das Variáveis Categóricas:")
        display(df_categoric.describe())

    return df_numeric.describe()


def grafico_percentual_valores_ausentes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Plota um gráfico de barras com o percentual de valores ausentes por variável
    e retorna uma tabela com a volumetria, total de nulos e percentual de nulos.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados.

    Retorna:
    pd.DataFrame: Tabela contendo as variáveis, volumetria, total de nulos e percentual de nulos.
    """
    # Cálculo dos valores ausentes
    total_nulos = df.isnull().sum()
    volumetria = len(df)
    perc_nulos = (total_nulos / volumetria) * 100

    # Criando DataFrame com as métricas
    tabela_nulos = pd.DataFrame({
        "Variável": df.columns,
        "Volumetria": volumetria,
        "Total nulos": total_nulos,
        "perc_nulos": perc_nulos
    })

    # Filtrando apenas as variáveis que possuem valores ausentes
    tabela_nulos = tabela_nulos[tabela_nulos["Total nulos"] > 0].sort_values(
        "perc_nulos", ascending=False)

    # Se não houver valores ausentes, retorna a mensagem e a tabela vazia
    if tabela_nulos.empty:
        print("✅ Nenhuma variável possui valores ausentes.")
        return tabela_nulos

    # Plotando o gráfico
    plt.figure(figsize=(10, 5))
    sns.barplot(x=tabela_nulos["Variável"],
                y=tabela_nulos["perc_nulos"], palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Percentual de valores ausentes (%)")
    plt.xlabel("Variáveis")
    plt.title("Percentual de Valores Ausentes por Variável")

    # Exibir os valores acima das barras
    for index, value in enumerate(tabela_nulos["perc_nulos"]):
        plt.text(index, value, f"{value:.1f}%",
                 ha="center", va="bottom", fontsize=8)

    plt.show()

    return tabela_nulos


def matriz_correlacao(df: pd.DataFrame) -> None:
    """
    Plota uma matriz de correlação de Pearson para variáveis numéricas.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados.
    """
    df_numeric = df.select_dtypes(include=[np.number])

    if df_numeric.empty:
        print("⚠️ Nenhuma variável numérica para calcular correlação.")
        return

    plt.figure(figsize=(10, 6))
    sns.heatmap(df_numeric.corr(), annot=True,
                cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Matriz de Correlação de Pearson")
    plt.show()


def histograma_variaveis_numericas(df: pd.DataFrame) -> None:
    """
    Plota histogramas para todas as variáveis numéricas do DataFrame.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados.
    """
    df_numeric = df.select_dtypes(include=[np.number])

    if df_numeric.empty:
        print("⚠️ Nenhuma variável numérica encontrada no DataFrame.")
        return

    df_numeric.hist(figsize=(8, 8), bins=20,
                    color="skyblue", edgecolor="black")
    plt.suptitle("Distribuição das Variáveis Numéricas", fontsize=14)
    plt.show()


def grafico_variaveis_categoricas(df: pd.DataFrame) -> None:
    """
    Plota gráficos de barras para as variáveis categóricas do DataFrame.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados.
    """
    df_categoric = df.select_dtypes(include=["O"])

    if df_categoric.empty:
        print("⚠️ Nenhuma variável categórica encontrada no DataFrame.")
        return

    for col in df_categoric.columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(
            x=df[col], order=df[col].value_counts().index, palette="Set2")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Distribuição da Variável Categórica: {col}")
        plt.ylabel("Contagem")
        plt.show()


################################# MODELAGEM - TREINAMENTO E ESCORAGEM ####################################################################


def treinar_e_tunar_modelo(target: str, data, model_name: str, setup_params: dict, tune_params: dict, problem_type="classification"):
    """
    Configura o ambiente do PyCaret, treina e realiza o tuning do modelo escolhido.

    Parâmetros:
    - target (str): Nome da coluna alvo.
    - data (DataFrame): Conjunto de dados para treinar o modelo.
    - model_name (str): Nome do modelo a ser criado (ex: 'rf' para Random Forest).
    - setup_params (dict): Parâmetros para a função setup() do PyCaret.
    - tune_params (dict): Parâmetros para a função tune_model() do PyCaret.
    - problem_type (str): Tipo de problema ('classification', 'regression' ou 'clustering').

    Retorna:
    - Modelo ajustado (tuned_model).
    """
    # Escolhe o tipo de problema
    if problem_type == "classification":
        cls_setup(data=data, target=target, **setup_params)
    elif problem_type == "regression":
        reg_setup(data=data, target=target, **setup_params)
    elif problem_type == "clustering":
        clu_setup(data=data, **setup_params)
    else:
        raise ValueError(
            "Tipo de problema inválido! Escolha entre 'classification', 'regression' ou 'clustering'.")

    # Criando o modelo
    model = create_model(model_name)

    # tunando o modelo
    tuned_model = tune_model(model, custom_grid=tune_params, optimize='AUC')

    return tuned_model


############################################ FUNCOES DE METRICAS E AVALIACAO DE MODELOS ############################################


def plot_comparacao_roc(bases_nomeadas: dict, nome_graficos: dict) -> None:
    """
    Gera gráficos de curva ROC dinamicamente de acordo com as bases fornecidas.

    Parâmetros:
    - bases_nomeadas (dict): Dicionário onde cada chave é o nome da base, e o valor é uma lista com:
        - DataFrame contendo colunas "y" (rótulos verdadeiros) e "score_1" (probabilidades preditas).
        - Número inteiro indicando o gráfico onde a curva deve ser plotada.
    - nome_graficos (dict): Dicionário que mapeia números inteiros para nomes dos gráficos.
    """

    # Identificar o número total de gráficos necessários
    num_graficos = max(v[1] for v in bases_nomeadas.values())

    # Criar a figura com subgráficos dinâmicos
    fig, axes = plt.subplots(1, num_graficos, figsize=(6 * num_graficos, 6))
    if num_graficos == 1:
        # Garantir que axes seja iterável quando há apenas um gráfico
        axes = [axes]

    # Dicionário para rastrear curvas por gráfico
    legendas_adicionadas = {i: False for i in range(1, num_graficos + 1)}

    # Iterar sobre os dataframes fornecidos
    for nome, (df, grafico_id) in bases_nomeadas.items():
        if df is not None and grafico_id in range(1, num_graficos + 1):
            ax = axes[grafico_id - 1]

            # Obter valores reais e probabilidades preditas
            y_true = df["y"]
            scores = df["score_1"]

            # Calcular curva ROC
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc_score = auc(fpr, tpr)

            # Plotar curva ROC
            ax.plot(fpr, tpr, label=f'{nome} (AUC = {auc_score:.2f})')

            # Configurações do gráfico
            if not legendas_adicionadas[grafico_id]:
                ax.plot([0, 1], [0, 1], linestyle='--',
                        color='gray')  # Linha diagonal
                titulo = nome_graficos.get(grafico_id, f'Gráfico {grafico_id}')
                ax.set_title(f'Curva ROC - {titulo}')
                ax.set_xlabel('Taxa de Falsos Positivos (FPR)')
                ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR)')
                legendas_adicionadas[grafico_id] = True

            ax.legend()
            ax.grid(True)

    # Ajustar layout e exibir gráficos
    plt.tight_layout()
    plt.show()


def plot_comparacao_prc(bases_nomeadas: dict, nome_graficos: dict) -> None:
    """
    Gera gráficos de Curva de Precisão-Revocação (PRC) dinamicamente para diferentes modelos,
    garantindo que todos fiquem lado a lado em uma única linha.

    Parâmetros:
    - bases_nomeadas (dict): Dicionário contendo os DataFrames nomeados, no formato:
        {
            "Nome da base": [dataframe, número_do_gráfico],
            ...
        }
    - nome_graficos (dict): Dicionário que mapeia os números dos gráficos para seus títulos:
        {
            número_do_gráfico: "Nome do Modelo",
            ...
        }

    Retorno:
    - None. A função exibe os gráficos.
    """
    # Identificar quantos gráficos são necessários
    num_graficos = len(set(num for _, num in bases_nomeadas.values()))

    # Forçar todos os gráficos em uma única linha
    colunas = num_graficos
    linhas = 1

    # Ajustar o tamanho da figura proporcionalmente ao número de gráficos
    fig, axes = plt.subplots(linhas, colunas, figsize=(
        5 * colunas, 5))  # Ajusta largura conforme necessidade

    # Garantir que axes seja iterável
    axes = axes.flatten() if num_graficos > 1 else [axes]

    # Organizar os DataFrames por gráfico
    dados_por_grafico = {i: [] for i in nome_graficos.keys()}
    for nome, (df, grafico_id) in bases_nomeadas.items():
        if grafico_id in dados_por_grafico:
            dados_por_grafico[grafico_id].append((nome, df))

    # Plotar cada gráfico
    for idx, (grafico_id, dados) in enumerate(dados_por_grafico.items()):
        ax = axes[idx]
        ax.set_title(f'Curva PRC - {nome_graficos[grafico_id]}')

        for nome_df, df in dados:
            y_true = df["y"]
            scores = df["score_1"]
            precision, recall, _ = precision_recall_curve(y_true, scores)
            auc_score = auc(recall, precision)
            ax.plot(recall, precision,
                    label=f'{nome_df} (AUC = {auc_score:.2f})')

        ax.set_xlabel('Revocação')
        ax.set_ylabel('Precisão')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_comparacao_ks(bases_nomeadas: dict, nome_graficos: dict) -> None:
    """
    Gera gráficos da Curva KS (Kolmogorov-Smirnov) para múltiplos modelos.

    Parâmetros:
    - bases_nomeadas (dict): Dicionário onde a chave é o nome da base, 
      e o valor é uma lista contendo [DataFrame, número do gráfico].
    - nome_graficos (dict): Dicionário onde a chave é o número do gráfico 
      e o valor é o nome do modelo correspondente.

    Retorno:
    - None. A função exibe os gráficos dinamicamente conforme a necessidade.
    """

    def calcular_ks(y_true, scores):
        """Calcula a curva KS para 'mau' (y=1) e 'bom' (y=0)."""
        df = pd.DataFrame({"y": y_true, "score": scores}
                          ).sort_values("score", ascending=True)

        total_mau = (df["y"] == 1).sum()
        total_bom = (df["y"] == 0).sum()

        df["cumulativo_mau"] = (df["y"] == 1).cumsum() / total_mau
        df["cumulativo_bom"] = (df["y"] == 0).cumsum() / total_bom

        df["ks"] = np.abs(df["cumulativo_mau"] - df["cumulativo_bom"])
        ks_max = df["ks"].max()
        ks_max_score = df.loc[df["ks"].idxmax(), "score"]
        probabilidade_mau_ks = df.loc[df["ks"].idxmax(), "cumulativo_mau"]

        return df["score"], df["cumulativo_mau"], df["cumulativo_bom"], ks_max, ks_max_score, probabilidade_mau_ks

    # Organizar os dataframes por gráficos
    graficos_dict = {}
    for nome, (df, num_grafico) in bases_nomeadas.items():
        if num_grafico not in graficos_dict:
            graficos_dict[num_grafico] = []
        graficos_dict[num_grafico].append((nome, df))

    num_graficos = len(graficos_dict)

    # Forçar no máximo 4 gráficos lado a lado
    cols = min(num_graficos, 4)
    rows = math.ceil(num_graficos / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    if num_graficos == 1:
        axes = [axes]  # Garantir que seja iterável mesmo com 1 gráfico
    elif rows == 1:
        axes = axes.reshape(1, -1)  # Ajuste para uma única linha

    axes = axes.flatten()  # Converter matriz para lista iterável

    # Definir cores fixas para manter a consistência
    cores_fixas = {
        "Treino": "blue",
        "Teste": "red",
        "OOT": "green"
    }

    # Plotar cada gráfico
    for idx, (num_grafico, bases) in enumerate(graficos_dict.items()):
        ax = axes[idx]
        titulo = nome_graficos.get(num_grafico, f"Gráfico {num_grafico}")

        for nome_base, df in bases:
            y = df["y"]
            scores = df["score_1"]

            prob, cum_mau, cum_bom, ks, ks_score, prob_mau = calcular_ks(
                y, scores)

            # Identificar se é treino, teste ou OOT pelo nome
            cor = cores_fixas["Treino"] if "Train" in nome_base else (
                cores_fixas["Teste"] if "Test" in nome_base else cores_fixas["OOT"]
            )

            ax.plot(prob, cum_mau, label=f'{nome_base} - Mau', color=cor)
            ax.plot(prob, cum_bom, linestyle='--',
                    label=f'{nome_base} - Bom', color=cor)
            ax.scatter(ks_score, prob_mau, color=cor, marker='o',
                       label=f'KS = {ks:.2%} (P={ks_score:.2f})')

        ax.set_title(f'Curva KS - {titulo}')
        ax.set_xlabel('Probabilidade de Mau')
        ax.set_ylabel('Percentual Acumulado da População')
        ax.legend()
        ax.grid(True)

    # Esconder gráficos vazios caso não sejam múltiplos perfeitos
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_comparacao_decil(bases_nomeadas: dict, nome_graficos: dict, num_divisoes: int = 10) -> None:
    """
    Gera gráficos comparativos da distribuição por decis para os modelos especificados no dicionário de entrada.

    Parâmetros:
    - bases_nomeadas (dict): Dicionário contendo os dataframes e o número do gráfico correspondente.
    - nome_graficos (dict): Dicionário mapeando os números dos gráficos para seus títulos.
    - num_divisoes (int): Número de divisões para os grupos (exemplo: 10 para decis, 5 para quintis etc.).

    Retorno:
    - None. A função exibe os gráficos.
    """

    def calcular_decil(y_true, scores, num_divisoes):
        """Divide os dados em grupos e calcula a taxa de eventos (mau) por grupo."""
        df = pd.DataFrame({"y": y_true, "score": scores})
        df["grupo"] = pd.qcut(df["score"], q=num_divisoes,
                              labels=False, duplicates="drop")
        return df.groupby("grupo")["y"].mean()

    # Identificar quantos gráficos únicos são necessários
    modelos_ids = set(v[1] for v in bases_nomeadas.values())
    num_graficos = len(modelos_ids)
    num_colunas = min(4, num_graficos)  # Máximo de 4 gráficos por linha
    # Definir o número de linhas
    num_linhas = math.ceil(num_graficos / num_colunas)

    fig, axes = plt.subplots(num_linhas, num_colunas, figsize=(
        num_colunas * 5, num_linhas * 5), squeeze=False)

    # Definir cores fixas para os tipos de dados
    cores = {"Treino": "blue", "Teste": "red", "OOT": "green"}

    # Organizar os DataFrames por gráfico
    dados_por_grafico = {k: [] for k in modelos_ids}
    for nome, (df, grafico_id) in bases_nomeadas.items():
        dados_por_grafico[grafico_id].append((nome, df))

    # Garantir que todos os gráficos tenham o mesmo range no eixo X e Y
    indices_padrao = np.arange(num_divisoes)
    max_y = 0  # Inicializar valor máximo do eixo Y

    # Primeiro loop: Encontrar o maior valor para padronizar o eixo Y
    for dados in dados_por_grafico.values():
        for _, df in dados:
            y_true = df["y"]
            scores = df["score_1"]
            decil = calcular_decil(y_true, scores, num_divisoes)
            # Atualiza o maior valor encontrado
            max_y = max(max_y, decil.max())

    # Segundo loop: Plotar os gráficos com eixo Y padronizado
    for i, (grafico_id, dados) in enumerate(dados_por_grafico.items()):
        linha, coluna = divmod(i, num_colunas)
        ax = axes[linha, coluna]

        largura = 0.3  # Largura das barras para separação
        deslocamento = -largura * (len(dados) / 2)  # Centralizar as barras

        for nome, df in dados:
            y_true = df["y"]
            scores = df["score_1"]
            decil = calcular_decil(y_true, scores, num_divisoes)
            taxa_mau = round(y_true.mean(), 4)

            # Identificar se é treino, teste ou OOT
            if "Train" in nome:
                tipo = "Treino"
            elif "Test" in nome:
                tipo = "Teste"
            else:
                tipo = "OOT"

            cor = cores[tipo]

            ax.bar(indices_padrao + deslocamento, decil,
                   largura, label=nome, color=cor)
            ax.axhline(taxa_mau, color=cor, linestyle="--",
                       linewidth=1, label=f"Média {tipo}: {taxa_mau:.2%}")

            deslocamento += largura  # Deslocar para a próxima barra

        # Aplicar eixo Y padronizado
        # Deixar 10% de margem para melhor visualização
        ax.set_ylim(0, max_y * 1.1)

        ax.set_title(f'Distribuição por Grupo - {nome_graficos[grafico_id]}')
        ax.set_xlabel(f'Grupo ({num_divisoes} divisões)')
        ax.set_ylabel('Taxa de Mau (%)')
        ax.set_xticks(indices_padrao)
        ax.set_xticklabels(indices_padrao + 1)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


def plot_comparacao_lift(bases_nomeadas: dict, nome_graficos: dict, n_bins: int = 10) -> None:
    """
    Gera gráficos da Curva Lift para múltiplos modelos.

    Parâmetros:
    - bases_nomeadas (dict): Dicionário onde a chave é o nome da base, 
      e o valor é uma lista contendo [DataFrame, número do gráfico].
    - nome_graficos (dict): Dicionário onde a chave é o número do gráfico 
      e o valor é o nome do modelo correspondente.
    - n_bins (int): Número de bins (percentis) para dividir os scores.

    Retorno:
    - None. A função exibe os gráficos dinamicamente conforme a necessidade.
    """

    def calcular_lift(y_true, scores, n_bins):
        """Calcula a Curva Lift agrupando os scores em percentis categóricos."""
        df = pd.DataFrame({"y": y_true, "score": scores})

        # Criar bins categóricos
        df["bin"] = pd.qcut(df["score"], q=n_bins, labels=[
                            f"{i+1}" for i in range(n_bins)], duplicates="drop")

        # Calcular a taxa de resposta por bin
        lift_df = df.groupby("bin").agg(
            total=("y", "count"),
            positivos=("y", "sum")
        ).reset_index()

        # Calcular a taxa de resposta acumulada
        lift_df["taxa_positivos"] = lift_df["positivos"] / lift_df["total"]
        taxa_media_positivos = df["y"].mean()

        # Calcular Lift
        lift_df["lift"] = lift_df["taxa_positivos"] / taxa_media_positivos

        return lift_df["bin"], lift_df["lift"]

    # Organizar os dataframes por gráficos
    graficos_dict = {}
    for nome, (df, num_grafico) in bases_nomeadas.items():
        if num_grafico not in graficos_dict:
            graficos_dict[num_grafico] = []
        graficos_dict[num_grafico].append((nome, df))

    num_graficos = len(graficos_dict)

    # Definir layout de gráficos dinâmico
    cols = min(num_graficos, 4)
    rows = math.ceil(num_graficos / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    if num_graficos == 1:
        axes = [axes]  # Garantir que seja iterável mesmo com 1 gráfico
    elif rows == 1:
        axes = axes.reshape(1, -1)  # Ajuste para uma única linha

    axes = axes.flatten()  # Converter matriz para lista iterável

    # Definir cores fixas para manter a consistência
    cores_fixas = {
        "Treino": "blue",
        "Teste": "red",
        "OOT": "green"
    }

    # Criar os gráficos
    for idx, (num_grafico, bases) in enumerate(graficos_dict.items()):
        ax = axes[idx]
        titulo = nome_graficos.get(num_grafico, f"Gráfico {num_grafico}")

        for nome_base, df in bases:
            y = df["y"]
            scores = df["score_1"]

            bins, lift = calcular_lift(y, scores, n_bins)

            # Identificar se é treino, teste ou OOT pelo nome
            cor = cores_fixas["Treino"] if "Train" in nome_base else (
                cores_fixas["Teste"] if "Test" in nome_base else cores_fixas["OOT"]
            )

            ax.plot(bins, lift, marker='o', linestyle='-',
                    label=f'{nome_base}', color=cor)

        ax.axhline(y=1, color="black", linestyle="--",
                   label="Baseline (Aleatório)")
        ax.set_title(f'Curva Lift - {titulo}')
        ax.set_xlabel('Bin')
        ax.set_ylabel('Lift')
        ax.legend()
        ax.grid(True)

        # Ajustar eixo X para ser categórico
        ax.set_xticks(range(n_bins))
        ax.set_xticklabels(bins, rotation=45)

    # Esconder gráficos vazios caso não sejam múltiplos perfeitos
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def gerar_tabela_avaliacao(
    train_lightgbm: pd.DataFrame,
    test_lightgbm: pd.DataFrame,
    train_regressao: pd.DataFrame,
    test_regressao: pd.DataFrame,
    test_oot_lightgbm: pd.DataFrame = None,
    test_oot_regressao: pd.DataFrame = None,
    num_divisoes: int = 10
) -> pd.DataFrame:
    """
    Gera uma tabela única contendo estatísticas sobre quantis de score para diferentes conjuntos de dados.

    Parâmetros:
    -----------
    train_lightgbm : pd.DataFrame
        Conjunto de treino do modelo LightGBM.
    test_lightgbm : pd.DataFrame
        Conjunto de teste do modelo LightGBM.
    train_regressao : pd.DataFrame
        Conjunto de treino do modelo de regressão.
    test_regressao : pd.DataFrame
        Conjunto de teste do modelo de regressão.
    test_oot_lightgbm : pd.DataFrame, opcional
        Conjunto de teste OOT (Out-of-Time) para o modelo LightGBM.
    test_oot_regressao : pd.DataFrame, opcional
        Conjunto de teste OOT (Out-of-Time) para o modelo de regressão.
    num_divisoes : int, padrão=10
        Número de quantis a serem calculados.

    Retorno:
    --------
    pd.DataFrame
        DataFrame contendo estatísticas agregadas por quantil.
    """

    # Lista de DataFrames para processamento
    dataframes = {
        "Train LightGBM": train_lightgbm,
        "Test LightGBM": test_lightgbm,
        "Train Regressão": train_regressao,
        "Test Regressão": test_regressao,
    }

    # Adiciona os OOT caso existam
    if test_oot_lightgbm is not None:
        dataframes["OOT LightGBM"] = test_oot_lightgbm
    if test_oot_regressao is not None:
        dataframes["OOT Regressão"] = test_oot_regressao

    # Lista para armazenar os resultados
    resultados = []

    for nome, df in dataframes.items():
        # Ordena pelo score_1 para garantir a correta distribuição dos quantis
        df = df.copy()
        df["quantil"] = pd.qcut(
            df["score_1"], num_divisoes, labels=False, duplicates='drop')

        # Ajusta para que os quantis comecem em 1 em vez de 0
        df["quantil"] += 1

        # Calcula métricas por quantil
        resumo = df.groupby("quantil").agg(
            total_casos=("id", "count"),
            total_mau=("y", "sum")
        ).reset_index()

        # Adiciona colunas derivadas
        resumo["total_bom"] = resumo["total_casos"] - resumo["total_mau"]
        resumo["maus_acumulados"] = resumo["total_mau"].cumsum()
        resumo["percentual_eventos"] = (
            resumo["total_mau"] / resumo["total_mau"].sum()) * 100
        resumo["cumulative_percentual_eventos"] = resumo["maus_acumulados"] / \
            resumo["total_mau"].sum() * 100

        # Cálculo do Gain
        resumo["gain"] = resumo["cumulative_percentual_eventos"]

        # Cálculo do Cumulative Lift
        resumo["cumulative_lift"] = resumo["gain"] / \
            ((resumo["quantil"] / num_divisoes) * 100)

        # Adiciona o nome do dataframe ao resultado
        resumo.insert(0, "nome_dataframe", nome)

        # Renomeia a coluna do quantil para melhor interpretação
        resumo.rename(columns={"quantil": "quantil", "total_casos": "total_casos", "total_mau": "total_mau",
                               "maus_acumulados": "maus_acumulados", "percentual_eventos": "% maus acumulados",
                               "gain": "Gain", "cumulative_lift": "Cumulative Lift"}, inplace=True)

        # Adiciona ao conjunto de resultados
        resultados.append(resumo)

    # Concatena todos os resultados em um único DataFrame
    resultado_final = pd.concat(resultados, ignore_index=True)

    resultado_final = resultado_final[['nome_dataframe', 'quantil', 'total_casos', 'total_mau', 'total_bom',
                                       'maus_acumulados', '% maus acumulados']]

    return resultado_final


def gerar_tabela_avaliacao(
    train_lightgbm: pd.DataFrame,
    test_lightgbm: pd.DataFrame,
    train_regressao: pd.DataFrame,
    test_regressao: pd.DataFrame,
    test_oot_lightgbm: pd.DataFrame = None,
    test_oot_regressao: pd.DataFrame = None,
    num_divisoes: int = 10
) -> pd.DataFrame:
    """
    Gera uma tabela única contendo estatísticas sobre quantis de score para diferentes conjuntos de dados.

    Parâmetros:
    -----------
    train_lightgbm : pd.DataFrame
        Conjunto de treino do modelo LightGBM.
    test_lightgbm : pd.DataFrame
        Conjunto de teste do modelo LightGBM.
    train_regressao : pd.DataFrame
        Conjunto de treino do modelo de regressão.
    test_regressao : pd.DataFrame
        Conjunto de teste do modelo de regressão.
    test_oot_lightgbm : pd.DataFrame, opcional
        Conjunto de teste OOT (Out-of-Time) para o modelo LightGBM.
    test_oot_regressao : pd.DataFrame, opcional
        Conjunto de teste OOT (Out-of-Time) para o modelo de regressão.
    num_divisoes : int, padrão=10
        Número de quantis a serem calculados.

    Retorno:
    --------
    pd.DataFrame
        DataFrame contendo estatísticas agregadas por quantil.
    """

    # Lista de DataFrames para processamento
    dataframes = {
        "Train LightGBM": train_lightgbm,
        "Test LightGBM": test_lightgbm,
        "Train Regressão": train_regressao,
        "Test Regressão": test_regressao,
    }

    # Adiciona os OOT caso existam
    if test_oot_lightgbm is not None:
        dataframes["OOT LightGBM"] = test_oot_lightgbm
    if test_oot_regressao is not None:
        dataframes["OOT Regressão"] = test_oot_regressao

    # Lista para armazenar os resultados
    resultados = []

    for nome, df in dataframes.items():
        # Ordena pelo score_1 para garantir a correta distribuição dos quantis
        df = df.copy()
        df["quantil"] = pd.qcut(
            df["score_1"], num_divisoes, labels=False, duplicates='drop')

        # Ajusta para que os quantis comecem em 1 em vez de 0
        df["quantil"] += 1

        # Calcula métricas por quantil
        resumo = df.groupby("quantil").agg(
            score_0_min=("score_0", "min"),
            score_0_max=("score_0", "max"),
            total_casos=("id", "count"),
            total_mau=("y", "sum")
        ).reset_index()

        # Adiciona colunas derivadas
        resumo["total_bom"] = resumo["total_casos"] - resumo["total_mau"]
        resumo["maus_acumulados"] = resumo["total_mau"].cumsum()
        resumo["% maus acumulados"] = (
            resumo["maus_acumulados"] / resumo["total_mau"].sum()) * 100

        # Cálculo do KS por faixa
        resumo["bons_acumulados"] = resumo["total_bom"].cumsum()
        resumo["% bons acumulados"] = (
            resumo["bons_acumulados"] / resumo["total_bom"].sum()) * 100
        resumo["KS"] = abs(resumo["% maus acumulados"] -
                           resumo["% bons acumulados"])

        # Adiciona o nome do dataframe ao resultado
        resumo.insert(0, "nome_dataframe", nome)

        # Renomeia a coluna do quantil para melhor interpretação
        resumo.rename(columns={"quantil": "quantil", "score_0_min": "score_0 min",
                      "score_0_max": "score_0 max"}, inplace=True)

        # Seleciona apenas as colunas desejadas
        resumo = resumo[['nome_dataframe', 'quantil', 'score_0 min', 'score_0 max', 'total_casos', 'total_mau', 'total_bom',
                         'maus_acumulados', '% maus acumulados', 'KS']]

        # Adiciona ao conjunto de resultados
        resultados.append(resumo)

    # Concatena todos os resultados em um único DataFrame
    resultado_final = pd.concat(resultados, ignore_index=True)

    return resultado_final


def calcular_ks_por_safra(base_escorada: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o valor máximo da estatística KS (Kolmogorov-Smirnov) para cada safra em um DataFrame.

    Parâmetros:
    base_escorada (pd.DataFrame): DataFrame contendo as colunas ['id', 'safra', 'y', 'score_1', 'score_0'].

    Retorna:
    pd.DataFrame: DataFrame com as colunas ['safra', 'contagem_de_linhas', 'ks_max', 'ponto_ks'].
    """

    def calcular_ks(df: pd.DataFrame) -> Tuple[float, float]:
        """
        Calcula o KS máximo de um DataFrame contendo colunas:
        ['id', 'safra', 'y', 'score_1', 'score_0'].

        Retorna o KS máximo (em percentual) e o ponto onde ele ocorre.
        """
        df = df.sort_values(
            by='score_1', ascending=False)  # Ordena pelo score_1 em ordem decrescente

        total_eventos = df['y'].sum()
        total_nao_eventos = (df['y'] == 0).sum()

        # Evita divisão por zero
        if total_eventos == 0 or total_nao_eventos == 0:
            return 0.0, np.nan

        df['acumulado_eventos'] = df['y'].cumsum() / total_eventos
        df['acumulado_nao_eventos'] = (
            (df['y'] == 0).cumsum()) / total_nao_eventos

        df['diferença'] = abs(df['acumulado_eventos'] -
                              df['acumulado_nao_eventos'])

        ks_max = df['diferença'].max() * 100  # Convertendo KS para percentual

        # Garantindo que ponto_ks seja um único valor
        ponto_ks = df.loc[df['diferença'] == df['diferença'].max(), 'score_1']
        ponto_ks = ponto_ks.iloc[0] if not np.isscalar(ponto_ks) else ponto_ks

        return ks_max, ponto_ks

    resultados = []

    for safra, grupo in base_escorada.groupby('safra', observed=True):
        ks_max, ponto_ks = calcular_ks(grupo)
        resultados.append([safra, len(grupo), ks_max, ponto_ks])

    tabela_resultados = pd.DataFrame(
        resultados, columns=['safra', 'contagem_de_linhas', 'ks_max', 'ponto_ks'])

    # Garantir que a safra seja ordenada corretamente como categoria
    tabela_resultados['safra'] = pd.Categorical(
        tabela_resultados['safra'], ordered=True)
    tabela_resultados = tabela_resultados.sort_values(by='safra')

    return tabela_resultados


def plotar_ks_safra(tabela_resultados: pd.DataFrame) -> None:
    """
    Gera um gráfico de barras mostrando a volumetria por safra e,
    no eixo secundário, o valor do KS por safra (em percentual).

    Parâmetros:
    tabela_resultados (pd.DataFrame): DataFrame com as colunas ['safra', 'contagem_de_linhas', 'ks_max', 'ponto_ks'].
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Gráfico de barras para volumetria
    ax1.bar(tabela_resultados['safra'].astype(str), tabela_resultados['contagem_de_linhas'],
            color='skyblue', label='Volumetria por Safra')
    ax1.set_xlabel('Safra')
    ax1.set_ylabel('Contagem de Linhas', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Criar um segundo eixo para o KS
    ax2 = ax1.twinx()
    ax2.plot(tabela_resultados['safra'].astype(str), tabela_resultados['ks_max'],
             color='red', marker='o', label='KS Máximo')
    ax2.set_ylabel('KS Máximo (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 100)  # Garantindo que o eixo do KS vá de 0 a 100

    # Título e legenda
    plt.title('Volumetria por Safra e KS Máximo (%)')
    fig.tight_layout()
    plt.show()


def calcular_psi(safra_referencia: pd.Series, safra_atual: pd.Series, bins: int = 10) -> Tuple[float, pd.DataFrame]:
    """
    Calcula o Population Stability Index (PSI) para uma variável contínua.

    Parâmetros:
    - safra_referencia (pd.Series): Série de valores da safra de referência.
    - safra_atual (pd.Series): Série de valores da safra atual.
    - bins (int): Número de faixas para discretizar os dados (padrão=10).

    Retorna:
    - Tuple[float, pd.DataFrame]: PSI total e um DataFrame com os detalhes por bin.
    """

    # Criar bins baseados na safra de referência
    bins_edges = np.linspace(safra_referencia.min(),
                             safra_referencia.max(), bins + 1)

    # Contar os valores dentro de cada bin
    ref_counts, _ = np.histogram(safra_referencia, bins=bins_edges)
    atual_counts, _ = np.histogram(safra_atual, bins=bins_edges)

    # Converter para proporções
    ref_props = ref_counts / ref_counts.sum()
    atual_props = atual_counts / atual_counts.sum()

    # Evitar divisão por zero (substituir 0 por um valor mínimo)
    ref_props = np.where(ref_props == 0, 0.0001, ref_props)
    atual_props = np.where(atual_props == 0, 0.0001, atual_props)

    # Calcular PSI para cada bin
    psi_values = (ref_props - atual_props) * np.log(ref_props / atual_props)

    # PSI total
    psi_total = psi_values.sum()

    # Criar DataFrame com os resultados
    psi_df = pd.DataFrame({
        'Bin': [f'{round(bins_edges[i], 2)} - {round(bins_edges[i+1], 2)}' for i in range(bins)],
        'Ref_Proporção': ref_props,
        'Atual_Proporção': atual_props,
        'PSI_Bin': psi_values
    })

    return psi_total, psi_df


def monitorar_variaveis_continuas(
    safra_referencia: pd.DataFrame, safra_atual: pd.DataFrame, colunas_numericas: List[str],
    psi_threshold: float = 0.1, ks_threshold: float = 0.05
) -> Dict[str, Dict[str, float]]:
    """
    Monitora a estabilidade das variáveis contínuas entre duas safras usando PSI e KS Test.

    Parâmetros:
    - safra_referencia (pd.DataFrame): DataFrame com os dados da safra de referência.
    - safra_atual (pd.DataFrame): DataFrame com os dados da safra atual.
    - colunas_numericas (List[str]): Lista de colunas numéricas a serem monitoradas.
    - psi_threshold (float): Limiar para considerar PSI significativo (padrão=0.1).
    - ks_threshold (float): Limiar para considerar KS Test significativo (padrão=0.05).

    Retorna:
    - Dict[str, Dict[str, float]]: Dicionário com variáveis que tiveram mudanças significativas.
    """

    alertas = {'psi': {}, 'ks': {}}

    for col in colunas_numericas:
        # Calcular PSI
        psi_total, _ = calcular_psi(safra_referencia[col], safra_atual[col])

        # Aplicar KS Test
        stat, p_value = ks_2samp(
            safra_referencia[col].dropna(), safra_atual[col].dropna())

        # Verificar se o PSI indica mudança significativa
        if psi_total >= psi_threshold:
            alertas['psi'][col] = {
                'PSI': psi_total,
                'Alerta': 'Mudança Moderada' if 0.1 <= psi_total < 0.25 else 'Mudança Significativa'
            }

        # Verificar se o KS Test indica mudança significativa
        if p_value < ks_threshold:
            alertas['ks'][col] = {
                'KS_Stat': stat,
                'p_value': p_value,
                'Alerta': 'Mudança Significativa'
            }

    psi = pd.DataFrame.from_dict(alertas['psi'], orient='index')

    ks = pd.DataFrame.from_dict(alertas['ks'], orient='index')

    return psi, ks


def obter_importancia_variaveis(modelo, nome_modelo=""):
    """
    Obtém a importância das variáveis de um modelo PyCaret.

    Parâmetros:
    modelo: Modelo treinado pelo PyCaret.
    nome_modelo: Nome do modelo (opcional, apenas para identificação).

    Retorna:
    DataFrame com as colunas 'nome_variavel' e 'importancia', ordenado do maior para o menor.
    """
    if hasattr(modelo, 'feature_importances_'):  # LightGBM e outros modelos baseados em árvores
        importancia = modelo.feature_importances_
        variaveis = modelo.feature_name_ if hasattr(
            modelo, 'feature_name_') else range(len(importancia))

    elif hasattr(modelo, 'coef_'):  # Modelos lineares como regressão logística
        importancia = modelo.coef_.ravel()  # Mantendo os valores originais dos betas
        variaveis = modelo.feature_names_in_ if hasattr(
            modelo, 'feature_names_in_') else range(len(importancia))

        # Criar DataFrame e ordenar pelo valor absoluto dos coeficientes, mas mantendo os sinais originais
        df_importancia = pd.DataFrame(
            {'nome_variavel': variaveis, 'importancia': importancia})
        df_importancia['importancia_abs'] = df_importancia['importancia'].abs()
        df_importancia = df_importancia.sort_values(by="importancia_abs", ascending=False).drop(
            columns=['importancia_abs']).reset_index(drop=True)

        return df_importancia

    else:
        raise ValueError(
            f"O modelo {nome_modelo} não possui um método de importância de variáveis.")

    df_importancia = pd.DataFrame(
        {'nome_variavel': variaveis, 'importancia': importancia})
    df_importancia = df_importancia.sort_values(
        by="importancia", ascending=False).reset_index(drop=True)

    return df_importancia


def calcular_metricas_multiplas(bases_escoradas: Dict[str, pd.DataFrame], limiar: float = 0.5) -> pd.DataFrame:
    """
    Calcula métricas de avaliação para um dicionário de DataFrames contendo bases escoradas.

    Parâmetros:
    -----------
    bases_escoradas : dict[str, pd.DataFrame]
        Dicionário onde:
        - As chaves são os nomes das bases.
        - Os valores são DataFrames com as colunas:
            - 'id': Identificador único.
            - 'safra': Período de referência.
            - 'y': Variável alvo (0 ou 1).
            - 'score_1': Probabilidade prevista da classe positiva.
            - 'score_0': Probabilidade prevista da classe negativa.

    limiar : float, opcional (default=0.5)
        Valor de corte para classificar as previsões. Valores acima do limiar são considerados positivos.

    Retorna:
    --------
    pd.DataFrame:
        DataFrame contendo as métricas para cada base no dicionário:
        - Nome da Base
        - Acurácia
        - Precisão
        - Recall
        - F1-score
        - AUC (Área sob a curva ROC)
        - KS MAX (Kolmogorov-Smirnov)
        - GINI
        - Verdadeiros Positivos (TP)
        - Falsos Positivos (FP)
        - Verdadeiros Negativos (TN)
        - Falsos Negativos (FN)
    """

    # Lista para armazenar os resultados
    resultados = []

    # Percorre cada DataFrame no dicionário
    for nome_base, base_escorada in bases_escoradas.items():
        # Verifica se o elemento é realmente um DataFrame
        if not isinstance(base_escorada, pd.DataFrame):
            raise TypeError(
                f"O valor associado a '{nome_base}' não é um DataFrame. Recebido: {type(base_escorada)}")

        # Verifica se as colunas necessárias estão presentes
        colunas_necessarias = {'id', 'safra', 'y', 'score_1', 'score_0'}
        if not colunas_necessarias.issubset(base_escorada.columns):
            raise ValueError(
                f"O DataFrame '{nome_base}' deve conter as colunas {colunas_necessarias}")

        # Obtendo os valores reais (y) e as previsões baseadas no limiar
        y_true = base_escorada['y']
        y_pred = (base_escorada['score_1'] >= limiar).astype(int)
        # Probabilidades da classe positiva
        y_scores = base_escorada['score_1']

        # Calculando métricas básicas
        acuracia = accuracy_score(y_true, y_pred)
        precisao = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Matriz de confusão
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Cálculo do AUC (Área sob a curva ROC)
        auc = roc_auc_score(y_true, y_scores)

        # KS MAX (Kolmogorov-Smirnov)
        ks_stat = ks_2samp(y_scores[y_true == 1],
                           y_scores[y_true == 0]).statistic

        # GINI = 2 * AUC - 1
        gini = 2 * auc - 1

        # Adiciona os resultados na lista
        resultados.append({
            "Nome da Base": nome_base,
            "Acurácia": round(acuracia, 4),
            "Precisão": round(precisao, 4),
            "Recall": round(recall, 4),
            "F1-score": round(f1, 4),
            "AUC": round(auc, 4),
            "KS MAX": round(ks_stat, 4),
            "GINI": round(gini, 4),
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn
        })

    # Converte a lista de resultados para um DataFrame e retorna
    return pd.DataFrame(resultados)


def train_woe_binning(
    df: pd.DataFrame,
    variables: List[str],
    target: str,
    time_variable: str,
    n_clusters: int = 4,
    plot_vars: List[str] = None
) -> Tuple[pd.DataFrame, Dict]:

    df_result = df.copy()
    df_result[time_variable] = df_result[time_variable].astype(
        str)  # Garantindo que safra seja categórica
    binning_rules = {}

    for variable in variables:
        # 1️⃣ Treinando o binning de WOE
        binning = OptimalBinning(name=variable, dtype="numerical", solver="cp")
        binning.fit(df[variable], df[target])

        # Aplicando o binning para obter os bins
        df_result[f"{variable}_bin"] = binning.transform(
            df[variable], metric="bins")

        # 2️⃣ Criando DataFrame para análise de estabilidade
        stability_df = df_result.groupby(
            [time_variable, f"{variable}_bin"]).size().unstack().fillna(0)
        stability_df = stability_df.div(stability_df.sum(axis=1), axis=0)

        # Ordenando `safra` para manter a sequência correta no gráfico
        stability_df = stability_df.sort_index()

        # 🚨 Ajustando o número de clusters para não exceder a quantidade de bins
        num_bins = len(stability_df.columns)
        # Garante que n_clusters não seja maior que o número de bins
        adjusted_clusters = min(n_clusters, num_bins)

        if adjusted_clusters < 2:
            print(
                f"⚠️ Aviso: {variable} tem apenas {num_bins} bins. Não será clusterizada.")
            bin_map = {bin_label: "Bin_1" for bin_label in stability_df.columns}
        else:
            # 3️⃣ Clusterizando bins com comportamento semelhante
            kmeans = KMeans(n_clusters=adjusted_clusters,
                            random_state=42, n_init=10)
            bin_clusters = kmeans.fit_predict(stability_df.T)
            bin_map = {bin_label: f"Bin_{cluster+1}" for bin_label,
                       cluster in zip(stability_df.columns, bin_clusters)}

        df_result[f"{variable}_bin_group"] = df_result[f"{variable}_bin"].map(
            bin_map)

        # 4️⃣ Criando as regras de transformação
        bin_edges = binning.splits
        bin_labels = sorted(
            df_result[f"{variable}_bin"].dropna().unique())  # Removendo NaNs
        bin_cluster_map = {bin_: bin_map[bin_] for bin_ in bin_labels}

        binning_rules[variable] = {
            "edges": bin_edges,
            "labels": bin_labels,
            "bin_to_group": bin_cluster_map
        }

        # 5️⃣ Plotando estabilidade antes e depois do agrupamento
        if plot_vars is None or variable in plot_vars:
            stability_grouped_df = df_result.groupby(
                [time_variable, f"{variable}_bin_group"]).size().unstack().fillna(0)
            stability_grouped_df = stability_grouped_df.div(
                stability_grouped_df.sum(axis=1), axis=0)

            # Ordenando a safra para melhor visualização
            stability_grouped_df = stability_grouped_df.sort_index()

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            stability_df.plot(ax=axes[0], marker='o')
            axes[0].set_title(
                f"Distribuição de {variable} ao Longo do Tempo (Antes do Agrupamento)")
            axes[0].set_ylabel("Proporção")
            axes[0].set_xlabel(time_variable)

            stability_grouped_df.plot(ax=axes[1], marker='o', cmap="tab10")
            axes[1].set_title(
                f"Distribuição de {variable} ao Longo do Tempo (Depois do Agrupamento)")
            axes[1].set_ylabel("Proporção")
            axes[1].set_xlabel(time_variable)

            plt.tight_layout()
            plt.show()

    return df_result, binning_rules


def escorar_modelo(base_treino: pd.DataFrame, base_dados_escorar: pd.DataFrame, caminho_modelo: str) -> pd.DataFrame:
    """
    Carrega um modelo salvo no PyCaret, faz previsões na base de dados fornecida e retorna os resultados.

    Parâmetros:
    -----------
    base_dados : pd.DataFrame
        Base de dados contendo as variáveis preditoras.

    caminho_modelo : str
        Caminho do arquivo do modelo salvo pelo PyCaret (.pkl).

    Retorno:
    --------
    pd.DataFrame
        DataFrame original com uma nova coluna contendo as previsões do modelo.
    """

    # Imputando missing
    regra_imputacao = escolher_estrategia_imputacao(base_treino)
    base_treino, regra_imputacao, dict_mediana, dict_media = aplicar_imputacao_treino(
        base_treino, regra_imputacao)
    base_dados_escorar = aplicar_imputacao_teste(
        base_dados_escorar, regra_imputacao, dict_mediana, dict_media)

    # 1. Carregar o modelo treinado e salvo anteriormente
    modelo = load_model(caminho_modelo)
    print(f"Modelo '{caminho_modelo}' carregado com sucesso!")

    # 2. Verificar se a base contém todas as colunas que o modelo espera
    colunas_modelo = modelo.feature_names_in_
    colunas_faltantes = [
        col for col in colunas_modelo if col not in base_dados_escorar.columns]

    if colunas_faltantes:
        raise ValueError(
            f"A base de dados fornecida não contém as colunas necessárias: {colunas_faltantes}")

    # 3. Fazer a predição na base de dados
    resultado = predict_model(
        modelo, data=base_dados_escorar, probability_threshold=0.3, raw_score=True)

    resultado = resultado[['id', 'safra', 'y', 'prediction_score_0', 'prediction_score_1']].rename(
        columns={'prediction_score_0': 'score_0', 'prediction_score_1': 'score_1'})

    # 4. Retornar a base original com a nova coluna de previsão
    return resultado


def plot_matriz_confusao(dataframes_dict, limiar=0.5, perc=False):
    """
    Plota as matrizes de confusão de cada dataframe fornecido no dicionário de entrada.

    Parâmetros:
    - dataframes_dict (dict): Dicionário contendo nomes como chave e DataFrame como valor.
    - limiar (float): Valor de corte para classificar score_1 como positivo.
    - perc (bool): Se True, exibe os valores da matriz de confusão em percentual.

    Retorna:
    - Exibe os gráficos das matrizes de confusão.
    """
    num_graphs = len(dataframes_dict)
    max_cols = 4  # Máximo de gráficos por linha
    # Calcula quantas linhas são necessárias
    rows = math.ceil(num_graphs / max_cols)

    fig, axes = plt.subplots(rows, min(max_cols, num_graphs), figsize=(
        5 * min(max_cols, num_graphs), 5 * rows))

    if rows == 1:
        axes = np.array([axes]) if num_graphs == 1 else np.array(
            axes).reshape(1, -1)

    axes = axes.flatten()  # Transformando em vetor para iteração

    color_palette = "coolwarm"  # Define a paleta de cores fixa para todos os gráficos

    for idx, (name, df) in enumerate(dataframes_dict.items()):
        y_true = df['y']
        y_pred = (df['score_1'] >= limiar).astype(
            int)  # Classificação com base no limiar

        cm = confusion_matrix(y_true, y_pred)

        if perc:
            cm = cm.astype('float') / cm.sum() * \
                100  # Converte para percentual

        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt=".1f" if perc else "d",
                    cmap=color_palette, cbar=False, ax=ax)
        ax.set_title(f'Matriz de Confusão - {name}')
        ax.set_xlabel('Predito')
        ax.set_ylabel('Real')

    # Remove gráficos extras se houver menos de 4*N
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def calcular_ks_por_safra(base_escorada: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o valor máximo da estatística KS (Kolmogorov-Smirnov) para cada safra em um DataFrame.

    Parâmetros:
    base_escorada (pd.DataFrame): DataFrame contendo as colunas ['id', 'safra', 'y', 'score_1', 'score_0'].

    Retorna:
    pd.DataFrame: DataFrame com as colunas ['safra', 'contagem_de_linhas', 'ks_max', 'ponto_ks'].
    """

    def calcular_ks(df: pd.DataFrame) -> Tuple[float, float]:
        """
        Calcula o KS máximo de um DataFrame contendo colunas:
        ['id', 'safra', 'y', 'score_1', 'score_0'].

        Retorna o KS máximo (em percentual) e o ponto onde ele ocorre.
        """
        df = df.sort_values(
            by='score_1', ascending=False)  # Ordena pelo score_1 em ordem decrescente

        total_eventos = df['y'].sum()
        total_nao_eventos = (df['y'] == 0).sum()

        # Evita divisão por zero
        if total_eventos == 0 or total_nao_eventos == 0:
            return 0.0, np.nan

        df['acumulado_eventos'] = df['y'].cumsum() / total_eventos
        df['acumulado_nao_eventos'] = (
            (df['y'] == 0).cumsum()) / total_nao_eventos

        df['diferença'] = abs(df['acumulado_eventos'] -
                              df['acumulado_nao_eventos'])

        ks_max = df['diferença'].max() * 100  # Convertendo KS para percentual

        # Garantindo que ponto_ks seja um único valor
        ponto_ks = df.loc[df['diferença'] == df['diferença'].max(), 'score_1']
        ponto_ks = ponto_ks.iloc[0] if not np.isscalar(ponto_ks) else ponto_ks

        return ks_max, ponto_ks

    resultados = []

    for safra, grupo in base_escorada.groupby('safra', observed=True):
        ks_max, ponto_ks = calcular_ks(grupo)
        resultados.append([safra, len(grupo), ks_max, ponto_ks])

    tabela_resultados = pd.DataFrame(
        resultados, columns=['safra', 'contagem_de_linhas', 'ks_max', 'ponto_ks'])

    # Garantir que a safra seja ordenada corretamente como categoria
    tabela_resultados['safra'] = pd.Categorical(
        tabela_resultados['safra'], ordered=True)
    tabela_resultados = tabela_resultados.sort_values(by='safra')

    return tabela_resultados


def calcular_ks_para_multiplas_bases(bases_nomeadas: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calcula a estatística KS por safra para múltiplas bases de dados e as concatena em um único DataFrame.

    Parâmetros:
    bases_nomeadas (Dict[str, pd.DataFrame]): Dicionário contendo nomes das bases como chave e DataFrames como valores.

    Retorna:
    pd.DataFrame: DataFrame consolidado com colunas ['Base', 'safra', 'contagem_de_linhas', 'ks_max', 'ponto_ks'].
    """
    resultados_finais = []

    for nome_base, df in bases_nomeadas.items():
        df_resultado = calcular_ks_por_safra(df)  # Calcula KS por safra
        df_resultado['Base'] = nome_base  # Adiciona a coluna de identificação
        resultados_finais.append(df_resultado)  # Armazena o resultado

    # Concatena os resultados em um único DataFrame
    df_consolidado = pd.concat(resultados_finais, ignore_index=True)

    return df_consolidado


def plotar_ks_safra(tabela_resultados: pd.DataFrame) -> None:
    """
    Gera um gráfico de linhas mostrando a evolução do KS máximo por safra,
    separando as linhas por categoria da base.

    Parâmetros:
    tabela_resultados (pd.DataFrame): DataFrame com as colunas ['safra', 'contagem_de_linhas', 'ks_max', 'ponto_ks', 'Base'].
    """
    plt.figure(figsize=(13, 5))

    # Identificar categorias únicas da coluna 'Base'
    categorias_base = tabela_resultados['Base'].unique()

    # Criar gráfico de linhas para cada categoria da base
    for categoria in categorias_base:
        subset = tabela_resultados[tabela_resultados['Base'] == categoria]
        plt.plot(subset['safra'].astype(str), subset['ks_max'],
                 marker='o', label=f'Base: {categoria}')

    # Configurações do gráfico
    plt.xlabel('Safra')
    plt.ylabel('KS Máximo (%)')
    plt.ylim(0, 100)  # Garantindo que o eixo do KS vá de 0 a 100
    plt.title('Evolução do KS Máximo por Safra')
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_psi(expected, actual, bins=10):
    """
    Calcula o PSI entre duas distribuições.
    """
    min_val, max_val = min(expected), max(expected)
    bins = np.linspace(min_val, max_val, bins + 1)

    expected_dist, _ = np.histogram(expected, bins=bins)
    actual_dist, _ = np.histogram(actual, bins=bins)

    expected_dist = expected_dist / expected_dist.sum()
    actual_dist = actual_dist / actual_dist.sum()

    expected_dist = np.where(expected_dist == 0, 0.0001, expected_dist)
    actual_dist = np.where(actual_dist == 0, 0.0001, actual_dist)

    psi_values = (expected_dist - actual_dist) * \
        np.log(expected_dist / actual_dist)
    return psi_values.sum()


def plot_psi(train_df, test_df, variavel, bins=10):
    """
    Plota um gráfico de linhas mostrando o PSI de cada safra de teste em relação ao treino.
    """
    # Criar referência com todas as safras de treino combinadas
    train_values = train_df[variavel].values
    test_months = sorted(test_df["safra"].unique())

    psi_results = []

    # Adicionar ponto inicial da referência (PSI = 0)
    psi_results.append({"safra": "Safras Treino", "PSI": 0})

    # Calcular PSI de cada safra de teste em relação ao treino
    for month in test_months:
        test_values = test_df[test_df["safra"] == month][variavel].values
        psi = calculate_psi(train_values, test_values, bins)
        psi_results.append({"safra": month, "PSI": psi})

    # Converter para DataFrame
    df_psi = pd.DataFrame(psi_results)

    # Converter safra para string (evita TypeError)
    df_psi["safra"] = df_psi["safra"].astype(str)

    # Criar o gráfico de linhas
    plt.figure(figsize=(10, 5))
    sns.lineplot(x="safra", y="PSI", data=df_psi, marker="o", color="b")

    # Adicionar linhas de referência para alerta e crítico
    plt.axhline(0.1, color="orange", linestyle="--", label="Alerta PSI = 0.1")
    plt.axhline(0.25, color="red", linestyle="--", label="Crítico PSI = 0.25")

    # Ajustes no gráfico
    plt.title("PSI: Comparação das Safras de Teste com Treino")
    plt.ylabel("PSI")
    plt.xlabel("Safra")
    plt.legend()
    plt.xticks(rotation=45)

    # Exibir gráfico
    plt.show()


def distance_correlation(x, y):
    """Calcula a Distância Correlacional (dCor) entre duas variáveis x e y."""
    x, y = np.asarray(x), np.asarray(y)

    def distance_matrix(a):
        return squareform(pdist(a[:, None], metric='euclidean'))

    def centering_matrix(d):
        n = d.shape[0]
        row_mean = d.mean(axis=1, keepdims=True)
        col_mean = d.mean(axis=0, keepdims=True)
        total_mean = d.mean()
        return d - row_mean - col_mean + total_mean

    A, B = centering_matrix(distance_matrix(
        x)), centering_matrix(distance_matrix(y))

    dCovXY = np.sqrt(np.mean(A * B))
    dVarX = np.sqrt(np.mean(A * A))
    dVarY = np.sqrt(np.mean(B * B))

    return dCovXY / np.sqrt(dVarX * dVarY) if dVarX > 0 and dVarY > 0 else 0


def distance_correlation_matrix(df):
    """
    Calcula a matriz de Distância Correlacional (dCor) para todas as variáveis de um DataFrame
    e plota um heatmap.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo as variáveis numéricas.

    Retorna:
        pd.DataFrame: Matriz de distância correlacional.
    """
    cols = df.columns
    n = len(cols)
    dcor_matrix = np.zeros((n, n))

    # Calcula a distância correlacional para cada par de variáveis
    for i in range(n):
        for j in range(n):
            if i <= j:  # Evita cálculos redundantes
                dcor_matrix[i, j] = distance_correlation(
                    df[cols[i]], df[cols[j]])
                dcor_matrix[j, i] = dcor_matrix[i, j]  # Matriz simétrica

    dcor_df = pd.DataFrame(dcor_matrix, index=cols, columns=cols)

    # Plotando o heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(dcor_df, annot=True, cmap="coolwarm",
                fmt=".2f", linewidths=0.5)
    plt.title("Heatmap de Distância Correlacional (dCor)")
    plt.show()

    return dcor_df
