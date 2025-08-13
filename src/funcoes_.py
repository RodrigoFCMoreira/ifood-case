# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from typing import Dict, List, Tuple
from typing import Optional
import os
import json
import joblib
import lightgbm as lgb
from datetime import datetime
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def avaliar_modelo_roc(df_train: pd.DataFrame,
                       df_test: pd.DataFrame,
                       score_col: str,
                       target_col: str,
                       title: str = "ROC Curve - Train vs Test"):
    """
    Plota ROC e retorna métricas (AUC e Gini) para treino e teste.

    Parâmetros
    ----------
    df_train, df_test : DataFrames com as colunas de score e target
    score_col : nome da coluna de score (probabilidade do classe 1)
    target_col: nome da coluna target (binário: 0/1)
    title     : título do gráfico

    Retorno
    -------
    metrics : dict com AUC e Gini de train e test
    """

    # --------- validações leves ---------
    for name, df in [("train", df_train), ("test", df_test)]:
        if score_col not in df.columns or target_col not in df.columns:
            raise ValueError(f"[{name}] faltam colunas '{score_col}' ou '{target_col}'")
        # target binário?
        vals = set(pd.Series(df[target_col]).dropna().unique().tolist())
        if not vals.issubset({0, 1}):
            raise ValueError(f"[{name}] target '{target_col}' deve ser binário (0/1). Valores: {vals}")

    # --------- preparar dados ---------
    def _prep(df):
        y = df[target_col].astype(int).values
        s = pd.to_numeric(df[score_col], errors="coerce").values
        mask = ~np.isnan(s)
        y, s = y[mask], s[mask]
        # opcional: clipe scores para [0,1] se houver ruído numérico
        s = np.clip(s, 0, 1)
        return y, s

    y_tr, s_tr = _prep(df_train)
    y_te, s_te = _prep(df_test)

    # --------- métricas ---------
    auc_tr = roc_auc_score(y_tr, s_tr)
    auc_te = roc_auc_score(y_te, s_te)
    gini_tr = 2*auc_tr - 1
    gini_te = 2*auc_te - 1

    fpr_tr, tpr_tr, _ = roc_curve(y_tr, s_tr)
    fpr_te, tpr_te, _ = roc_curve(y_te, s_te)

    # --------- plot ---------
    plt.figure(figsize=(7, 5))
    plt.plot(fpr_tr, tpr_tr, label=f"Train AUC={auc_tr:.3f} | Gini={gini_tr:.3f}", lw=2)
    plt.plot(fpr_te, tpr_te, label=f"Test  AUC={auc_te:.3f} | Gini={gini_te:.3f}", lw=2, linestyle="--")
    plt.plot([0,1], [0,1], color="gray", lw=1, linestyle=":")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        "train": {"auc": float(auc_tr), "gini": float(gini_tr)},
        "test":  {"auc": float(auc_te), "gini": float(gini_te)}
    }
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def plot_ks_prob_side_by_side(df_train, df_test, score_col, target_col, title="KS Curve - Probabilidade"):
    """
    Plota curvas KS (eixo X = probabilidade prevista, 0→1) para Treino e Teste lado a lado.
    Retorna dict com KS e a probabilidade p* onde o KS é máximo em cada base.
    """

    def ks_frame_prob(df):
        d = df[[score_col, target_col]].copy()
        d[score_col] = pd.to_numeric(d[score_col], errors="coerce").clip(0, 1)
        d = d.dropna(subset=[score_col, target_col])

        # ordenar crescente p/ começar em 0
        d = d.sort_values(score_col, ascending=True).reset_index(drop=True)

        pos_tot = max(int((d[target_col] == 1).sum()), 1)
        neg_tot = max(int((d[target_col] == 0).sum()), 1)

        d["pos_acum"] = (d[target_col] == 1).cumsum() / pos_tot
        d["neg_acum"] = (d[target_col] == 0).cumsum() / neg_tot
        d["ks"] = (d["pos_acum"] - d["neg_acum"]).abs()

        ks_idx = int(d["ks"].values.argmax())
        ks_val = float(d.loc[ks_idx, "ks"])
        p_star = float(d.loc[ks_idx, score_col])                 # probabilidade onde KS é máximo
        pos_at_p = float(d.loc[ks_idx, "pos_acum"])
        neg_at_p = float(d.loc[ks_idx, "neg_acum"])
        return d, ks_val, p_star, pos_at_p, neg_at_p, ks_idx

    tr, ks_tr, p_tr, pos_tr, neg_tr, idx_tr = ks_frame_prob(df_train)
    te, ks_te, p_te, pos_te, neg_te, idx_te = ks_frame_prob(df_test)

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, d, ks_val, p_star, pos_at_p, neg_at_p, idx, nome in [
        (axes[0], tr, ks_tr, p_tr, pos_tr, neg_tr, idx_tr, "Treino"),
        (axes[1], te, ks_te, p_te, pos_te, neg_te, idx_te, "Teste"),
    ]:
        ax.plot(d[score_col], d["pos_acum"], label="Positivos acumulados", color="#1f77b4", lw=1.8)
        ax.plot(d[score_col], d["neg_acum"], label="Negativos acumulados", color="#d62728", lw=1.8)

        # linha vertical e marcação do ponto p*
        ax.axvline(x=p_star, color="#2ca02c", linestyle="--", lw=1.3, label=f"KS = {ks_val:.3f}")
        ax.scatter([p_star, p_star], [pos_at_p, neg_at_p], color=["#1f77b4","#d62728"], zorder=3)
        ax.text(p_star, 1.02, f"p*={p_star:.3f}", ha="center", va="bottom", fontsize=9, color="#2ca02c")

        ax.set_title(f"{nome} — KS: {ks_val:.3f}")
        ax.set_xlabel("Probabilidade prevista")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Acumulado")
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="lower right")
    fig.suptitle(title, fontsize=15)
    plt.tight_layout()
    plt.show()

    return {
        "train": {"KS": ks_tr, "prob_ks_max": p_tr, "pos_acum_at_p": pos_tr, "neg_acum_at_p": neg_tr},
        "test":  {"KS": ks_te, "prob_ks_max": p_te, "pos_acum_at_p": pos_te, "neg_acum_at_p": neg_te},
    }


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def avaliar_por_limiar(df_train, df_test, score_col, target_col, limiar=0.5):
    """
    Avalia métricas de classificação para um limiar específico no score.
    Retorna DataFrame com métricas para treino e teste.

    Parâmetros:
    -----------
    df_train, df_test : pd.DataFrame
        DataFrames contendo as colunas de score e target.
    score_col : str
        Nome da coluna com os scores (probabilidades preditas).
    target_col : str
        Nome da coluna com o target binário (0 ou 1).
    limiar : float
        Valor de corte entre 0 e 1 para classificar como positivo.

    Retorna:
    --------
    pd.DataFrame com métricas.
    """
    
    def calc_metrics(df, nome_base):
        y_true = df[target_col]
        y_score = df[score_col]
        y_pred = (y_score >= limiar).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_score)

        # KS
        df_temp = df[[score_col, target_col]].copy()
        df_temp = df_temp.sort_values(by=score_col, ascending=False)
        df_temp["cum_eventos"] = (df_temp[target_col] == 1).cumsum() / (df_temp[target_col] == 1).sum()
        df_temp["cum_nao_eventos"] = (df_temp[target_col] == 0).cumsum() / (df_temp[target_col] == 0).sum()
        ks = np.max(np.abs(df_temp["cum_eventos"] - df_temp["cum_nao_eventos"]))

        gini = 2 * auc - 1

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        return {
            "Nome da Base": nome_base,
            "Acurácia": acc,
            "Precisão": prec,
            "Recall": rec,
            "F1-score": f1,
            "AUC": auc,
            "KS MAX": ks,
            "GINI": gini,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn
        }
    
    resultados = []
    resultados.append(calc_metrics(df_train, "Treino"))
    resultados.append(calc_metrics(df_test, "Teste"))
    
    return pd.DataFrame(resultados)

def plot_confusion_from_metrics(metrics_df):
    """
    Plota matrizes de confusão para Treino e Teste lado a lado,
    usando os valores de TP, FP, TN, FN já presentes no dataframe de métricas.
    
    Parâmetros:
    - metrics_df: DataFrame com colunas ['Nome da Base','TP','FP','TN','FN']
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for i, ax in enumerate(axes):
        row = metrics_df.iloc[i]
        cm = [
            [row['TN'], row['FP']],
            [row['FN'], row['TP']]
        ]
        
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Prev. Negativo", "Prev. Positivo"],
            yticklabels=["Real Negativo", "Real Positivo"],
            ax=ax
        )
        ax.set_title(f"{row['Nome da Base']} - Limiar Atual")
        ax.set_xlabel("Previsão")
        ax.set_ylabel("Real")
    
    plt.tight_layout()
    plt.show()

# =========================
# Funções utilitárias
# =========================
# ---------- utilitários ----------

def _detectar_categoricas(X: pd.DataFrame):
    return X.select_dtypes(include=['object', 'category']).columns.tolist()

def _padronizar_categorias_para_fit(X: pd.DataFrame, categorical_cols: List[str]):
    X_conv = X.copy()
    categorias_map = {}
    for c in categorical_cols:
        if X_conv[c].dtype == 'object':
            X_conv[c] = X_conv[c].astype('category')
        elif X_conv[c].dtype.name != 'category':
            X_conv[c] = X_conv[c].astype('category')
        categorias_map[c] = X_conv[c].cat.categories.tolist()
    return X_conv, categorias_map

def _alinhar_colunas_para_predict(X: pd.DataFrame, features_treino: list):
    X_alinhado = X.copy()
    faltantes = [c for c in features_treino if c not in X_alinhado.columns]
    for c in faltantes:
        X_alinhado[c] = np.nan
    sobras = [c for c in X_alinhado.columns if c not in features_treino]
    if sobras:
        X_alinhado = X_alinhado.drop(columns=sobras)
    return X_alinhado[features_treino]

def _aplicar_categorias_salvas(X: pd.DataFrame, categorias_map: Dict[str, list]):
    X_conv = X.copy()
    for c, cats in categorias_map.items():
        if c in X_conv.columns:
            X_conv[c] = X_conv[c].astype('category')
            X_conv[c] = X_conv[c].cat.set_categories(cats)
    return X_conv

# ---------- 1) Treinar, escorar e salvar ----------
def treinar_e_salvar_modelo_lgbm(
    parametros_escolhidos: Dict,
    train_tratado: pd.DataFrame,   # base ORIGINAL (com IDs, explicativas e target)
    target: str,
    path: str,
    score_col: str = 'score_modelo',
    id_cols: List[str] = None      # <--- NOVO: colunas para manter no escore, mas excluir do treino
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Treina LGBMClassifier, gera importâncias, escora e salva artefatos.
    - 'id_cols' são mantidas no 'train_escorado' e excluídas do treino.
    Retorna: (importance_df, train_escorado_completo, modelo_pkl)
    """
    os.makedirs(path, exist_ok=True)
    id_cols = id_cols or []

    # separa X e y (features = tudo menos target e ids)
    features = [c for c in train_tratado.columns if c not in id_cols + [target]]
    X = train_tratado[features].copy()
    y = train_tratado[target].astype(int).values

    # categorias
    categorical_cols = _detectar_categoricas(X)
    X_fit, categorias_map = _padronizar_categorias_para_fit(X, categorical_cols)

    # modelo
    model = lgb.LGBMClassifier(**parametros_escolhidos)
    model.fit(X_fit, y, categorical_feature=categorical_cols)

    # importâncias
    booster = model.booster_
    importance_df = pd.DataFrame({
        'feature': booster.feature_name(),
        'gain': booster.feature_importance(importance_type='gain'),
        'split': booster.feature_importance(importance_type='split')
    }).sort_values('gain', ascending=False).reset_index(drop=True)

    # escora o TREINO nas MESMAS features
    proba = model.predict_proba(X_fit)[:, 1]

    # monta o train_escorado como a BASE ORIGINAL COMPLETA + score
    train_escorado = train_tratado.copy()
    train_escorado[score_col] = proba

    # salvar csvs
    importance_df.to_csv(os.path.join(path, 'feature_importance.csv'), index=False)
    train_escorado.to_csv(os.path.join(path, 'train_escorado.csv'), index=False)

    # metadados (salvos dentro do pkl)
    meta = {
        'features': features,
        'categorical_cols': categorical_cols,
        'categorias_map': categorias_map,
        'target': target,
        'score_col': score_col,
        'id_cols': id_cols,
        'parametros_escolhidos': parametros_escolhidos,
        'treinado_em': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(os.path.join(path, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump({k: v for k, v in meta.items() if k != 'categorias_map'}, f, ensure_ascii=False, indent=2)

    modelo_pkl = {'model': model, 'meta': meta}
    joblib.dump(modelo_pkl, os.path.join(path, 'model.pkl'))

    return importance_df, train_escorado, modelo_pkl

# ---------- 2) Carregar e aplicar no teste ----------
def aplicar_modelo_pkl(
    df_teste: pd.DataFrame,   # base ORIGINAL de teste (com IDs, só não precisa ter o target)
    path: str,
    score_col: str = 'score_modelo'
) -> pd.DataFrame:
    """
    Carrega model.pkl, alinha colunas/categorias e escora df_teste.
    Retorna df_teste COMPLETO + coluna 'score_col'.
    """
    bundle = joblib.load(os.path.join(path, 'model.pkl'))
    model = bundle['model']
    meta = bundle['meta']

    features_treino = meta['features']
    categorias_map = meta.get('categorias_map', {})
    categorical_cols = meta.get('categorical_cols', [])
    score_col = meta.get('score_col', score_col)

    # prepara X a partir do df_teste (mantendo df original para retorno)
    X = df_teste.copy()

    # garante dtype e ordem das features de treino
    for c in categorical_cols:
        if c in X.columns:
            if X[c].dtype == 'object':
                X[c] = X[c].astype('category')
            elif X[c].dtype.name != 'category':
                X[c] = X[c].astype('category')

    X = _alinhar_colunas_para_predict(X, features_treino)
    X = _aplicar_categorias_salvas(X, categorias_map)

    proba = model.predict_proba(X)[:, 1]

    df_escorado = df_teste.copy()
    df_escorado[score_col] = proba
    return df_escorado

def matriz_correlacao(df: pd.DataFrame) -> None:

    import seaborn as sns
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

def plot_safra_conversion_rate(
    df: pd.DataFrame,
    safra_col: str = "safra",
    conversao_col: str = "y",
    conv_rate_min: Optional[float] = None,
    conv_rate_max: Optional[float] = None
) -> pd.DataFrame:
    """
    Gera um gráfico de barras com a contagem por safra e
    uma linha com a taxa de conversão no eixo secundário.

    Retorna:
    - DataFrame com: safra, contagem, total_convertidos, total_nao_convertidos, conversion_rate.
    """
    import matplotlib.pyplot as plt
    from typing import Optional
    import pandas as pd

    # Garantir que safra seja numérica para ordenação
    df[safra_col] = pd.to_numeric(df[safra_col], errors="coerce")

    # Agrupar e ordenar
    safra_stats = (
        df.groupby(safra_col)
        .agg(
            contagem=(conversao_col, "count"),
            total_convertidos=(conversao_col, "sum")
        )
        .reset_index()
        .sort_values(safra_col)
    )

    # Calcular não convertidos e taxa
    safra_stats["total_nao_convertidos"] = safra_stats["contagem"] - safra_stats["total_convertidos"]
    safra_stats["conversion_rate"] = safra_stats["total_convertidos"] / safra_stats["contagem"]

    # Criar gráfico
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Barras - total
    ax1.bar(safra_stats[safra_col].astype(str), safra_stats["contagem"],
            color="blue", alpha=0.6, label="Total")
    ax1.set_xlabel("Safra")
    ax1.set_ylabel("Total de IDs", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_xticklabels(safra_stats[safra_col].astype(str), rotation=45)

    # Linha - taxa de conversão
    ax2 = ax1.twinx()
    ax2.plot(safra_stats[safra_col].astype(str), safra_stats["conversion_rate"],
             color="green", marker="o", linestyle="-", linewidth=2, label="Taxa de Conversão")
    ax2.set_ylabel("Taxa de Conversão (%)", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    # Limites opcionais do eixo secundário
    if conv_rate_min is not None and conv_rate_max is not None:
        ax2.set_ylim(conv_rate_min, conv_rate_max)

    plt.title("Total por Safra e Taxa de Conversão")
    fig.tight_layout()
    plt.show()

    return safra_stats


def grid_search_inteligente_lgbm_gini(
    df: pd.DataFrame,
    target: str,
    categorical_cols: List[str] = None,
    cv_splits: int = 5,
    early_stopping_rounds: int = 100,
    max_combinations: int = 60,   # limita busca para não explodir tempo
    random_state: int = 42,
    verbose: int = 1
) -> Tuple[Dict, pd.DataFrame]:
    """
    Faz um grid search 'inteligente' de hiperparâmetros do LightGBM otimizando o Gini (2*AUC - 1),
    com validação cruzada estratificada e early stopping por fold.

    Parâmetros
    ----------
    df : DataFrame com explicativas + target
    target : nome da coluna alvo (binária 0/1)
    categorical_cols : lista opcional de colunas categóricas (se None, detecta automaticamente)
    cv_splits : nº de folds (default 5)
    early_stopping_rounds : paciência do early stopping por fold
    max_combinations : nº máximo de combinações a avaliar (amostra se o grid exceder isso)
    random_state : semente
    verbose : 0 silencioso, 1 progresso básico

    Retorna
    -------
    best_params : dict com os melhores hiperparâmetros (inclui n_estimators recomendado)
    historico_df : DataFrame com resultados por combinação (gini_mean, gini_std, etc.)
    """

    rng = np.random.RandomState(random_state)

    # === 1) separa X, y e trata categorias ===
    features = [c for c in df.columns if c != target]
    X = df[features].copy()
    y = df[target].astype(int).values

    if categorical_cols is None:
        categorical_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
    # Converte objetos para category (LightGBM lida nativamente)
    for c in categorical_cols:
        if X[c].dtype == 'object':
            X[c] = X[c].astype('category')

    # === 2) base de parâmetros "fixos" e medidas de desbalanceamento ===
    pos = y.sum()
    neg = len(y) - pos
    base_params = {
        'objective': 'binary',        # modelo de classificação binária
        'boosting_type': 'gbdt',      # Gradient Boosted Decision Trees
        'random_state': random_state,
        'n_jobs': -1
    }
    # Se muito desbalanceado, usa scale_pos_weight ~ (neg/pos)
    if pos > 0 and neg / max(pos, 1) >= 3:
        base_params['scale_pos_weight'] = neg / max(pos, 1)

    # === 3) grade de hiperparâmetros (valores que "fazem sentido") ===
    # Comentários do porquê de cada um:
    param_grid = {
        # taxa de aprendizado: menor -> mais árvores, maior estabilidade; maior -> converge rápido, risco overfit
        'learning_rate': [0.01, 0.05, 0.1],
        # controle do tamanho das folhas: mais folhas capturam interações complexas, mas podem overfit
        'num_leaves': [31, 63, 127],
        # profundidade máxima: -1 = ilimitado; valores moderados ajudam a regularizar
        'max_depth': [-1, 6, 8, 10],
        # amostra mínima por folha: maior -> mais suave/regularizado
        'min_child_samples': [20, 50, 100],
        # amostragem de linhas por árvore (bagging): <1 ajuda a reduzir variância
        'subsample': [0.7, 0.9, 1.0],
        # amostragem de colunas por árvore: <1 ajuda a reduzir correlação entre árvores
        'colsample_bytree': [0.7, 0.9, 1.0],
        # L1 e L2, penalizações que ajudam a controlar complexidade
        'reg_alpha': [0.0, 0.1, 0.5],
        'reg_lambda': [0.0, 0.1, 0.5],
        # ganho mínimo para dividir (split) — aumenta a exigência para criar novas folhas
        'min_split_gain': [0.0, 0.1]
    }

    # Cria todas as combinações e filtra algumas inviáveis
    all_keys = list(param_grid.keys())
    all_values = list(param_grid.values())
    combos = []
    for values in itertools.product(*all_values):
        params = dict(zip(all_keys, values))
        # regra prática: se max_depth > 0, manter num_leaves <= 2**max_depth
        if params['max_depth'] > 0 and params['num_leaves'] > 2 ** params['max_depth']:
            continue
        combos.append(params)

    # Amostra se exceder max_combinations (mantém "inteligente" e viável)
    if len(combos) > max_combinations:
        idx = rng.choice(len(combos), size=max_combinations, replace=False)
        combos = [combos[i] for i in idx]

    if verbose:
        print(f"Total de combinações a avaliar: {len(combos)}")

    # === 4) validação cruzada com early stopping por fold ===
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    resultados = []
    best = {'gini_mean': -np.inf, 'params': None, 'best_iters': None}

    for i, params in enumerate(combos, 1):
        if verbose:
            print(f"[{i}/{len(combos)}] Testando params: {params}")

        ginis_fold = []
        best_iters = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            model = lgb.LGBMClassifier(
                n_estimators=5000,            # grande o suficiente; early stopping decide o melhor
                **base_params,
                **params
            )

            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric='auc',
                categorical_feature=categorical_cols,
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
            )

            # probabilidade positiva usando melhor iteração encontrada no fold
            y_pred = model.predict_proba(X_va, num_iteration=model.best_iteration_)[:, 1]
            auc = roc_auc_score(y_va, y_pred)
            gini = 2 * auc - 1
            ginis_fold.append(gini)
            best_iters.append(model.best_iteration_)

        gini_mean = float(np.mean(ginis_fold))
        gini_std = float(np.std(ginis_fold, ddof=1))

        resultados.append({
            **params,
            'gini_mean': gini_mean,
            'gini_std': gini_std,
            'best_iter_mean': int(np.round(np.mean(best_iters))),
            'best_iter_median': int(np.median(best_iters))
        })

        if gini_mean > best['gini_mean']:
            best = {
                'gini_mean': gini_mean,
                'params': params,
                'best_iters': best_iters
            }

    historico_df = pd.DataFrame(resultados).sort_values('gini_mean', ascending=False).reset_index(drop=True)

    # === 5) monta best_params final, sugerindo n_estimators ~ média das melhores iterações ===
    best_n_estimators = int(np.round(np.mean(best['best_iters']))) if best['best_iters'] else 500
    best_params = {
        **base_params,
        **best['params'],
        'n_estimators': best_n_estimators,
        # dica: ao treinar final, mantenha early stopping e um n_estimators maior (ex.: 2x)
        'sugestao_treino_final': {
            'n_estimators_sugerido': int(best_n_estimators * 2),
            'usar_early_stopping_rounds': early_stopping_rounds
        },
        'cv_gini_mean': best['gini_mean']
    }

    if verbose:
        print("\nMelhores hiperparâmetros (por Gini médio em CV):")
        print(best_params)

    return best_params, historico_df


def eliminacao_progressiva_por_importancia(
    df, colunas_id, target, 
    min_features=1, max_steps=None, 
    plot=True
):
    """
    Executa eliminação progressiva de variáveis usando a função
    modelar_lightgbm_feature_selection. Em cada etapa treina, mede KS/Gini,
    salva a lista de variáveis utilizadas e remove a variável de menor ganho.

    Retorna:
        resultados_df (pd.DataFrame) com colunas:
            ETAPA, QTD_VARIAVEL, GINI, KS, VARIAVEIS_EXPLICATIVAS
        figs (dict) com figuras de KS e Gini por etapa, se plot=True.
    """
    current_features = [c for c in df.columns if c not in colunas_id + [target]]
    
    resultados = []
    etapa = 0
    figs = {}

    while len(current_features) >= max(min_features, 1):
        if (max_steps is not None) and (etapa >= max_steps):
            break

        etapa += 1

        cols_rodada = colunas_id + [target] + current_features
        df_step = df[cols_rodada].copy()

        nova_lista, ks, gini = modelar_lightgbm_feature_selection(df_step, colunas_id, target)

        resultados.append({
            'ETAPA': etapa,
            'QTD_VARIAVEL': len(current_features),
            'GINI': gini,
            'KS': ks,
            'VARIAVEIS_EXPLICATIVAS': current_features.copy()
        })

        current_features = nova_lista
        if len(current_features) < max(min_features, 1):
            break

    resultados_df = pd.DataFrame(resultados)

    if plot and not resultados_df.empty:
        # KS por etapa
        fig_ks, ax_ks = plt.subplots(figsize=(7, 4))
        ax_ks.plot(resultados_df['ETAPA'], resultados_df['KS'], marker='o', color='#1f77b4')
        ax_ks.set_title('KS por Etapa')
        ax_ks.set_xlabel('Etapa')
        ax_ks.set_ylabel('KS')
        ax_ks.grid(True, alpha=0.3)
        for x, ks_val, q in zip(resultados_df['ETAPA'], resultados_df['KS'], resultados_df['QTD_VARIAVEL']):
            ax_ks.annotate(f'n={q}', (x, ks_val), textcoords="offset points", xytext=(0,8), ha='center', fontsize=8)

        # Gini por etapa
        fig_gini, ax_gini = plt.subplots(figsize=(7, 4))
        ax_gini.plot(resultados_df['ETAPA'], resultados_df['GINI'], marker='o', color='#d62728')
        ax_gini.set_title('Gini por Etapa')
        ax_gini.set_xlabel('Etapa')
        ax_gini.set_ylabel('Gini')
        ax_gini.grid(True, alpha=0.3)
        for x, g_val, q in zip(resultados_df['ETAPA'], resultados_df['GINI'], resultados_df['QTD_VARIAVEL']):
            ax_gini.annotate(f'n={q}', (x, g_val), textcoords="offset points", xytext=(0,8), ha='center', fontsize=8)

        figs = {'ks': fig_ks, 'gini': fig_gini}

    return resultados_df, figs


def modelar_lightgbm_feature_selection(df, colunas_id, target):
    features = [col for col in df.columns if col not in colunas_id + [target]]
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    categorical_features = X_train.select_dtypes(include=['category', 'object']).columns.tolist()

    lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
    lgb_test = lgb.Dataset(X_test, label=y_test, categorical_feature=categorical_features, reference=lgb_train)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'seed': 42
    }

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_test],
        num_boost_round=100,
        callbacks=[
            lgb.early_stopping(stopping_rounds=10, verbose=False)
        ]
    )

    importance_df = pd.DataFrame({
        'feature': model.feature_name(),
        'gain': model.feature_importance(importance_type='gain')
    }).sort_values(by='gain', ascending=False).reset_index(drop=True)

    menor_importancia = importance_df.iloc[-1]['feature']
    nova_lista_features = [f for f in features if f != menor_importancia]

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    df_ks = pd.DataFrame({'y': y_test, 'y_pred': y_pred})
    df_ks = df_ks.sort_values(by='y_pred', ascending=False)
    df_ks['cum_event'] = (df_ks['y'] == 1).cumsum()
    df_ks['cum_non_event'] = (df_ks['y'] == 0).cumsum()
    total_event = (df_ks['y'] == 1).sum()
    total_non_event = (df_ks['y'] == 0).sum()
    df_ks['tpr'] = df_ks['cum_event'] / total_event
    df_ks['fpr'] = df_ks['cum_non_event'] / total_non_event
    ks = max(df_ks['tpr'] - df_ks['fpr'])

    auc = roc_auc_score(y_test, y_pred)
    gini = 2 * auc - 1

    return nova_lista_features, ks, gini


def build_missing_plan():
    rows = [
        # ——— Chaves/target (sem imputação) ———
        ("ID","id","none","chave"), 
        ("cliente_id","id","none","chave"),
        ("offer_id","id","none","chave"),
        ("time_since_test_start","tempo","none","marcador temporal"),
        ("target_sucesso","target","none","rótulo"),

        # ——— Perfil ———
        ("idade","numérica_contínua","mediana","ausência informativa (criar flag_miss)"),
        ("genero","categórica","MISSING","não informado tem significado"),
        ("limite_credito","numérica_contínua","mediana","distribuição assimétrica"),
        ("safra_registro","categórica","MISSING","categoria faltante explícita"),
        ("historico_conversao","proporção","0","sem histórico ⇒ 0"),

        # ——— Recência & frequência (compras e ofertas) ———
        ("dias_desde_ultima_transacao","recência","-1","nunca ocorreu"),
        ("dias_desde_ultima_oferta_recebida","recência","-1","nunca ocorreu"),
        ("dias_desde_ultima_oferta_visualizada","recência","-1","nunca ocorreu"),
        ("media_tempo_entre_transacoes","numérica_contínua","-1","sem base histórica"),
        ("transacoes_distintas_dias","numérica_contínua","-1","sem base histórica"),

        ("qtd_ofertas_anteriores","contagem","0","nunca ocorreu"),
        ("qtd_ofertas_visualizadas","contagem","0","nunca ocorreu"),
        ("qtd_ofertas_completas_validas","contagem","0","nunca ocorreu"),
        ("qtd_transacoes_7d","contagem","0","nunca ocorreu"),
        ("qtd_transacoes_14d","contagem","0","nunca ocorreu"),
        ("qtd_ofertas_recebidas_14d","contagem","0","nunca ocorreu"),
        ("qtd_ofertas_visualizadas_14d","contagem","0","nunca ocorreu"),
        ("qtd_conversoes_validas_14d","contagem","0","nunca ocorreu"),
        ("canais_recebidos_total","contagem","0","nunca ocorreu"),
        ("canais_preferidos","contagem","0","nunca ocorreu"),

        # ——— Taxas/razões ———
        ("taxa_visualizacao","proporção","0","denominador ausente ⇒ 0"),
        ("taxa_conversao","proporção","0","denominador ausente ⇒ 0"),
        ("email_ratio_historico","proporção","0","histórico ausente ⇒ 0"),
        ("taxa_visualizacao_historica_tipo","proporção","0","histórico ausente ⇒ 0"),
        ("taxa_conversao_historica_tipo","proporção","0","histórico ausente ⇒ 0"),
        ("taxa_visualizacao_historica_oferta","proporção","0","histórico ausente ⇒ 0"),
        ("taxa_conversao_historica_canal_social","proporção","0","histórico ausente ⇒ 0"),
        ("afinidade_canais_conv","proporção","0","histórico ausente ⇒ 0"),
        ("taxa_conversao_historica_bucket_duracao","proporção","0","histórico ausente ⇒ 0"),
        ("taxa_conversao_historica_bucket_desconto","proporção","0","histórico ausente ⇒ 0"),
        ("taxa_conversao_historica_bucket_minimo","proporção","0","histórico ausente ⇒ 0"),

        # ——— Características da oferta ———
        ("offer_type","categórica","MISSING","categoria faltante explícita"),
        ("discount_value","inteira","mediana_int","campo do produto"),
        ("min_value","inteira","mediana_int","campo do produto"),
        ("duration","inteira","mediana_int","campo do produto"),

        # ——— Canais (flags atuais e histórico recebido) ———
        ("web","categórica","MISSING","flag de canal"),
        ("email","categórica","MISSING","flag de canal"),
        ("mobile","categórica","MISSING","flag de canal"),
        ("social","categórica","MISSING","flag de canal"),
        ("recebeu_web","categórica","MISSING","flag histórico"),
        ("recebeu_email","categórica","MISSING","flag histórico"),
        ("recebeu_mobile","categórica","MISSING","flag histórico"),
        ("recebeu_social","categórica","MISSING","flag histórico"),

        # ——— Cliente-oferta ———
        ("n_instancias_anteriores_mesma_oferta","contagem","0","nunca ocorreu"),
    ]
    return pd.DataFrame(rows, columns=["variavel","tipo","tratamento_missing","motivo"])

def fit_missing_prep(train: pd.DataFrame, plan: pd.DataFrame):
    """
    Aprende parâmetros necessários para imputação (ex.: medianas).
    Retorna um dicionário 'prep' com tudo que será usado no apply.
    """
    prep = {"medianas": {}, "medianas_int": {}, "categorias_add_missing": []}

    # Colunas por estratégia
    to_minus1 = plan.query("tratamento_missing == '-1'")["variavel"].tolist()
    to_zero   = plan.query("tratamento_missing == '0'")["variavel"].tolist()
    to_med    = plan.query("tratamento_missing == 'mediana'")["variavel"].tolist()
    to_medint = plan.query("tratamento_missing == 'mediana_int'")["variavel"].tolist()
    to_cat    = plan.query("tratamento_missing == 'MISSING'")["variavel"].tolist()

    # Medianas (float)
    for c in to_med:
        if c in train.columns:
            prep["medianas"][c] = float(train[c].median(skipna=True))

    # Medianas para inteiras (arredonda e guarda como int)
    for c in to_medint:
        if c in train.columns:
            med = train[c].median(skipna=True)
            med_int = int(np.round(med)) if pd.notna(med) else 0
            prep["medianas_int"][c] = med_int

    # Guardar listas
    prep["to_minus1"] = [c for c in to_minus1 if c in train.columns]
    prep["to_zero"]   = [c for c in to_zero   if c in train.columns]
    prep["to_med"]    = [c for c in to_med    if c in train.columns]
    prep["to_medint"] = [c for c in to_medint if c in train.columns]
    prep["to_cat"]    = [c for c in to_cat    if c in train.columns]

    # Checar categorias e marcar quais precisam incluir 'MISSING'
    for c in prep["to_cat"]:
        if str(train[c].dtype) == "category":
            # adicionaremos a categoria 'MISSING' no apply
            prep["categorias_add_missing"].append(c)

    return prep


def apply_missing_prep(df: pd.DataFrame, prep: dict, copy=True) -> pd.DataFrame:
    """
    Aplica a imputação de missing de acordo com 'prep' (treinado no train).
    Não altera dtypes de id/target; mantém categorias como category.
    """
    out = df.copy() if copy else df

    # -1 (recências / médias sem base)
    for c in prep.get("to_minus1", []):
        if c in out.columns:
            out[c] = out[c].fillna(-1)

    # 0 (contagens / proporções sem histórico)
    for c in prep.get("to_zero", []):
        if c in out.columns:
            out[c] = out[c].fillna(0)

    # mediana (float)
    for c, med in prep.get("medianas", {}).items():
        if c in out.columns:
            out[c] = out[c].fillna(med)

    # mediana_int (inteiros do produto)
    for c, med in prep.get("medianas_int", {}).items():
        if c in out.columns:
            out[c] = out[c].fillna(med).astype("int64")

    # categóricas: criar/garantir label 'MISSING' e preencher
    for c in prep.get("to_cat", []):
        if c in out.columns:
            if str(out[c].dtype) == "category":
                if c in prep.get("categorias_add_missing", []):
                    # incluir a categoria se não existir
                    if "MISSING" not in list(out[c].cat.categories):
                        out[c] = out[c].cat.add_categories(["MISSING"])
                out[c] = out[c].fillna("MISSING")
            else:
                # se por acaso não está como category, ainda assim preenche
                out[c] = out[c].astype("object").fillna("MISSING").astype("category")

    return out

# =========================
# Utilidades gerais
# =========================
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def detectar_tipos(
    df: pd.DataFrame,
    target: str | None = None,
    exclude_cols: list[str] | None = None,     # ex.: ["ID","cliente_id","offer_id","time_since_test_start"]
    force_num: list[str] | None = None,        # ex.: ["email","mobile","social","web", ...]
    force_cat: list[str] | None = None,        # ex.: ["genero","offer_type","safra_registro"]
    int_low_card_as_cat: bool = False,         # se True, inteiros de baixa cardinalidade viram categóricas
    max_card_cat: int = 30
):
    exclude_cols = set(exclude_cols or [])
    force_num = set(force_num or [])
    force_cat = set(force_cat or [])

    # ponto de partida
    num_cols = set(df.select_dtypes(include=[np.number]).columns)
    cat_cols = set(df.select_dtypes(include=["object","category","bool"]).columns)

    # remover exclusões e target de ambos
    for c in list(num_cols):
        if c in exclude_cols or (target and c == target):
            num_cols.discard(c)
    for c in list(cat_cols):
        if c in exclude_cols or (target and c == target):
            cat_cols.discard(c)

    # detectar binárias (0/1) e mantê-las como NUMÉRICAS
    def is_binary_series(s: pd.Series) -> bool:
        vals = set(pd.Series(s).dropna().unique().tolist())
        return len(vals) <= 2 and vals.issubset({0, 1})

    # mover qualquer binária que tenha ido pra cat -> num
    for c in list(cat_cols):
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and is_binary_series(df[c]):
            cat_cols.discard(c)
            num_cols.add(c)

    # inteiros de baixa cardinalidade: só vira categórica se int_low_card_as_cat=True
    if int_low_card_as_cat:
        int_cols = df.select_dtypes(include=["int8","int16","int32","int64","UInt8","UInt16","UInt32","UInt64"]).columns
        for c in int_cols:
            if c in num_cols and c not in exclude_cols and (target is None or c != target):
                if df[c].nunique(dropna=True) <= max_card_cat and not is_binary_series(df[c]):
                    num_cols.discard(c)
                    cat_cols.add(c)

    # aplicar forças manuais
    for c in force_num:
        cat_cols.discard(c)
        if c not in exclude_cols and (target is None or c != target):
            num_cols.add(c)
    for c in force_cat:
        num_cols.discard(c)
        if c not in exclude_cols and (target is None or c != target):
            cat_cols.add(c)

    # ordenar listas
    num_cols = sorted(num_cols)
    cat_cols = sorted(cat_cols)
    return num_cols, cat_cols

# =========================
# 1) Balanceamento do target
# =========================
def target_balance(df, target):
    """
    Distribuição do target (contagem e %).
    """
    cnt = df[target].value_counts(dropna=False)
    pct = df[target].value_counts(normalize=True, dropna=False).mul(100).round(2)
    out = pd.DataFrame({"count": cnt, "pct_%": pct}).sort_index()
    return out

# =========================
# 2) Missing por variável
# =========================
def resumo_missing(df):
    """
    % de missing por coluna, ordenado desc.
    """
    miss = df.isna().mean().mul(100).sort_values(ascending=False).round(2)
    return miss.to_frame("%_missing")

# =========================
# 3) Distribuições numéricas + separação por target
# =========================
def resumo_numericas(df, num_cols):
    """
    Estatísticas descritivas para colunas numéricas.
    """
    return df[num_cols].describe().T

def numericas_por_target(df, num_cols, target):
    agg = {c: ["mean", "median", "std", "min", "max"] for c in num_cols}
    out = df.groupby(target).agg(agg)  # colunas = MultiIndex (var, stat)
    out.columns = [f"{var}_{stat}" for var, stat in out.columns]  # <- aqui!
    return out  # linhas: classes do target; colunas: var_estat

# =========================
# 4) Categóricas x target (qui-quadrado)
# =========================
from scipy.stats import chi2_contingency

def chi2_categoricas(df, cat_cols, target, min_freq=10):
    """
    Teste de qui-quadrado para variáveis categóricas vs target.
    Retorna p-values (quanto menor, maior evidência de associação).
    - Filtra categorias raras agrupando-as em 'OTHER' quando necessário.
    """
    results = []
    for c in cat_cols:
        # tabela de contingência
        vc = df[c].astype("object").copy()
        # Agrupar categorias raras
        freq = vc.value_counts()
        raras = freq[freq < min_freq].index
        vc = vc.where(~vc.isin(raras), other="OTHER")
        tab = pd.crosstab(vc, df[target])
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue
        chi2, p, dof, exp = chi2_contingency(tab)
        results.append((c, p, dof, tab.sum().sum()))
    out = pd.DataFrame(results, columns=["variavel", "p_value", "dof", "n"])
    return out.sort_values("p_value")

# =========================
# 5) Correlação e pares fortes
# =========================
def correlacao_numericas(df, num_cols, metodo="pearson", lim=0.9):
    """
    Matriz de correlação + lista de pares com |corr| >= lim.
    """
    corr = df[num_cols].corr(method=metodo)
    pares = []
    for i, c1 in enumerate(num_cols):
        for c2 in num_cols[i+1:]:
            v = corr.loc[c1, c2]
            if pd.notna(v) and abs(v) >= lim:
                pares.append((c1, c2, v))
    pares_df = pd.DataFrame(pares, columns=["var1","var2","corr"])
    pares_df = pares_df.sort_values("corr", key=lambda s: s.abs(), ascending=False)
    return corr, pares_df

# =========================
# 6) VIF (multicolinearidade)
# =========================
def calcular_vif(df, num_cols, max_vars=40):
    """
    Calcula VIF (Variance Inflation Factor) usando statsmodels.
    Se statsmodels não estiver instalado, retorna None.
    Sugestão: aplique em subconjuntos (max_vars) para evitar custo alto.
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        import statsmodels.api as sm
    except Exception:
        return None

    cols = num_cols[:max_vars]
    X = df[cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = sm.add_constant(X, has_constant='add')
    vifs = []
    for i, c in enumerate(X.columns):
        if c == "const": 
            continue
        vifs.append((c, variance_inflation_factor(X.values, i)))
    out = pd.DataFrame(vifs, columns=["variavel","VIF"]).sort_values("VIF", ascending=False)
    return out

# =========================
# 7) WoE / IV (força preditiva univariada)
# =========================
def woe_iv(df, col, target, n_bins=10, metodo="quantile", eps=1e-6):
    """
    Calcula WoE/IV para uma variável (numérica ou categórica).
    - Para numérica: binning por quantil (default) ou cut (=bins iguais).
    - Para categórica: usa categorias como bins.
    Retorna (tabela_bins, IV_total).
    """
    s = df[col]
    y = df[target].astype(int)

    # Binning
    if pd.api.types.is_numeric_dtype(s):
        try:
            if metodo == "quantile":
                binned = pd.qcut(s, q=min(n_bins, s.nunique()), duplicates="drop")
            else:
                binned = pd.cut(s, bins=n_bins)
        except Exception:
            binned = s  # se der erro, segue sem bin
    else:
        binned = s.astype("object")

    tab = pd.crosstab(binned, y)
    # garantir colunas 0 e 1
    if 0 not in tab.columns: tab[0] = 0
    if 1 not in tab.columns: tab[1] = 0
    tab = tab[[0,1]].rename(columns={0:"neg", 1:"pos"})

    tab["tot"] = tab["neg"] + tab["pos"]
    N = tab["neg"].sum()
    P = tab["pos"].sum()

    tab["dist_neg"] = (tab["neg"] + eps) / (N + eps)
    tab["dist_pos"] = (tab["pos"] + eps) / (P + eps)
    tab["woe"] = np.log(tab["dist_pos"] / tab["dist_neg"])
    tab["iv_bin"] = (tab["dist_pos"] - tab["dist_neg"]) * tab["woe"]
    IV = tab["iv_bin"].sum()

    tab = tab.reset_index().rename(columns={col: "bin"})
    return tab, IV

def woe_iv_todas(df, cols, target, **kwargs):
    """
    Calcula IV para uma lista de colunas. Retorna ranking por IV desc.
    """
    rows = []
    for c in cols:
        try:
            _, iv = woe_iv(df, c, target, **kwargs)
            rows.append((c, iv))
        except Exception as e:
            rows.append((c, np.nan))
    out = pd.DataFrame(rows, columns=["variavel","IV"]).sort_values("IV", ascending=False)
    return out

# =========================
# 8) Importância inicial (baseline) com RF
# =========================
def baseline_importance_rf(df, feats, target, n_estimators=400, random_state=42, cv=5):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    import pandas as pd

    X = df[feats].copy()
    y = df[target].astype(int).copy()

    # 1) Se alguma numérica virou object por acidente, tenta reconverter
    for c in X.columns:
        if X[c].dtype == "object":
            # tenta coagir para número; se não der, fica object (categórica)
            coerced = pd.to_numeric(X[c], errors="coerce")
            # se boa parte virou número, adotamos a versão numérica
            if coerced.notna().mean() > 0.8:
                X[c] = coerced

    # 2) Imputar numéricas com mediana
    num_mask = X.dtypes.apply(lambda t: np.issubdtype(t, np.number))
    num_cols = X.columns[num_mask]
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())

    # 3) Encode simples nas categóricas (resto)
    cat_cols = X.columns[~num_mask]
    for c in cat_cols:
        le = LabelEncoder()
        X[c] = X[c].astype("str").fillna("__MISSING__")
        X[c] = le.fit_transform(X[c])

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    importancias = np.zeros(len(feats))
    aucs = []

    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state
        )
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_va)[:, 1]
        aucs.append(roc_auc_score(y_va, p))
        importancias += model.feature_importances_

    importancias /= cv
    imp_df = (pd.DataFrame({"variavel": feats, "importancia": importancias})
                .sort_values("importancia", ascending=False))
    return imp_df, float(np.mean(aucs)), float(np.std(aucs))


# =========================
# 9) Interações candidatas (tabelas de conversão)
# =========================
def taxa_conversao_cruzada(df, col_a, col_b, target, min_freq=50):
    """
    Tabela de taxas de conversão para pares (col_a x col_b).
    Filtra combinações raras (menos de min_freq).
    """
    tab = pd.pivot_table(df, index=col_a, columns=col_b, values=target, aggfunc=["mean","count"])
    # filtrar por frequência
    cnt = tab["count"]
    mean = tab["mean"].where(cnt >= min_freq)
    return mean  # matriz de taxas de conversão

# =========================
# 10) Variância quase nula (remover ruído)
# =========================
def quase_constantes(df, cols, thresh=0.99):
    """
    Retorna variáveis cuja categoria/modalidade mais comum >= thresh (ex.: 99% igual).
    Útil para descartar variáveis pouco informativas.
    """
    out = []
    for c in cols:
        top_frac = df[c].value_counts(normalize=True, dropna=False).max()
        if top_frac >= thresh:
            out.append((c, round(top_frac,4)))
    return pd.DataFrame(out, columns=["variavel","frac_modal"]).sort_values("frac_modal", ascending=False)

def perfil_base_conversao(
    base_modelo: pd.DataFrame,
    id_col: str,
    target_col: str,
    safra_col: str
) -> dict:
    """
    Calcula métricas básicas do perfil da base de dados focada em conversão.

    Parâmetros:
    - base_modelo (pd.DataFrame): DataFrame contendo os dados a serem analisados.
    - id_col (str): Nome da coluna que representa o identificador único (ID).
    - target_col (str): Nome da coluna que representa a variável alvo (1 = convertido, 0 = não convertido).
    - safra_col (str): Nome da coluna que representa a safra.

    Retorna:
    - dict: Dicionário contendo:
        - shape: Tupla com a quantidade de linhas e colunas.
        - tipos_variaveis: Contagem dos tipos das variáveis.
        - ids_unicos: Quantidade de IDs únicos.
        - taxa_conversao: Taxa de conversão da base.
        - volumetria_safras: Quantidade de registros por safra.
    """
    perfil = {}

    # 1. Shape da base
    perfil['shape'] = f"Essa base possui {base_modelo.shape[0]} linhas e {base_modelo.shape[1]} colunas"

    # 2. Tipos das variáveis
    perfil['tipos_variaveis'] = base_modelo.dtypes.value_counts().to_dict()

    # 3. IDs únicos
    perfil['ids_unicos'] = base_modelo[id_col].nunique()

    # 4. Taxa de conversão
    if target_col in base_modelo.columns:
        total_conversoes = base_modelo[target_col].value_counts().to_dict()

        # Evitar KeyError se faltar alguma das classes
        nao_convertidos = total_conversoes.get(0, 0)
        convertidos = total_conversoes.get(1, 0)

        taxa_conv = convertidos / (nao_convertidos + convertidos) if (nao_convertidos + convertidos) > 0 else 0

        perfil['taxa_conversao'] = {
            "não_convertidos": nao_convertidos,
            "convertidos": convertidos,
            "taxa_percentual": round(taxa_conv * 100, 2)
        }
    else:
        perfil['taxa_conversao'] = "Coluna alvo não encontrada."

    # 5. Volumetria por safra
    if safra_col in base_modelo.columns:
        perfil['volumetria_safras'] = dict(
            sorted(base_modelo[safra_col].value_counts().to_dict().items())
        )
    else:
        perfil['volumetria_safras'] = "Coluna safra não encontrada."

    # Prints amigáveis
    print("📊 Perfil da base de dados (Conversão)")
    print(f"Shape da base: {perfil['shape']}")
    print(f"Tipos de variáveis: {perfil['tipos_variaveis']}")
    print(f"IDs únicos: {perfil['ids_unicos']}")
    if isinstance(perfil['taxa_conversao'], dict):
        print(f"Taxa de conversão: {perfil['taxa_conversao']['taxa_percentual']}% "
              f"({perfil['taxa_conversao']['convertidos']} convertidos, "
              f"{perfil['taxa_conversao']['não_convertidos']} não convertidos)")
    else:
        print(f"Taxa de conversão: {perfil['taxa_conversao']}")
    print(f"Volumetria das safras: {perfil['volumetria_safras']}")
    print("\n")

    return perfil

def aplicar_dtypes(df: pd.DataFrame, dtypes_dict: dict) -> pd.DataFrame:
    """
    Aplica conversão de tipos de colunas conforme dicionário {coluna: dtype}.
    
    Parâmetros:
    - df: DataFrame original
    - dtypes_dict: dicionário com {coluna: tipo_pandas}
    
    Retorna:
    - DataFrame com os tipos ajustados
    """
    for col, dtype in dtypes_dict.items():
        if col in df.columns:
            try:
                if dtype == "category":
                    df[col] = df[col].astype("category")
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype) if "float" in dtype or "int" in dtype else df[col].astype(dtype)
            except Exception as e:
                print(f"[AVISO] Não foi possível converter coluna '{col}' para {dtype}: {e}")
    return df
