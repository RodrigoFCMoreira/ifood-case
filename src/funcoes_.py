# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def build_missing_plan():
    rows = [
        # ——— Chaves/target (sem imputação) ———
        ("ID","id","none","chave"), 
        ("cliente_id","id","none","chave"),
        ("offer_id","id","none","chave"),
        ("time_since_test_start","tempo","none","marcador temporal"),
        ("target_sucesso","target","none","rótulo"),

        # ——— Perfil ———
        ("idade","numérica_contínua","-1","ausência informativa (criar flag_miss)"),
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
