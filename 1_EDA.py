# -*- coding: utf-8 -*-
"""Análise exploratória LIAR — convertido de 1_EDA.ipynb.
Instalação: pip install -r requirements_EDA.txt
Execução: python 1_EDA.py
"""

# # Análise Exploratória Inicial do Dataset LIAR

# ## Objetivo
#
# Carregamento, transformação dos dados e análide exploratória inicial dos dados sobre configurações de classes distintas: LIAR (6 classes), SentimentalLIAR (2 classes), BinaryLIAR (2 classes) e TernaryLIAR (3 classes).

# ## Declaração de uso de Inteligência Artificial
#
# O código e descrição foram gerados e adaptados com auxílio de IA (ChatGPT, OpenAI).

# ## Etapas implementadas
#
# **Carregamento dos dados**
#
# - Utilização do dataset LIAR como no trabalho [1], e disponibilizado pela Universidade da Califórnia, Santa Bárbara (UCSB) [2].
# - Adaptação do carregamento para utilizar os arquivos originais TSV, evitando incompatibilidades com versões recentes da biblioteca `datasets`.
# - Preservação das partições originais de treinamento, validação e teste.
#
# **Transformação dos rótulos**
#
# - LIAR: manutenção das seis classes originais.
# - SentimentalLIAR: agrupamento em duas classes, considerando `true` e `mostly-true` como verdadeiras, conforme o trabalho [3].
# - BinaryLIAR: agrupamento em duas classes, incluindo `half-true` na classe verdadeira, conforme feito em [4].
# - TernaryLIAR: agrupamento em três classes — falsas, parcialmente verdadeiras e verdadeiras, como adotado em [5].
# - Criação de um DataFrame contendo os textos e as quatro classificações numéricas para comparação.
#
# **Pré-processamento textual**
#
# - Tratamento de valores ausentes.
# - Conversão dos textos para minúsculas.
# - Remoção de URLs, números, pontuação e caracteres especiais.
# - Tokenização por espaços.
# - Remoção de stopwords em inglês e palavras com menos de três caracteres.
# - Reconstrução do texto processado para análise.
# O texto original foi preservado em uma coluna separada.
#
# **Visualização dos dados**
#
# - Geração de nuvens de palavras para cada classe dos quatro datasets.
# - Identificação das classes por número e nome.
# - Aplicação de cores distintas para facilitar a interpretação visual.
# - Organização das nuvens lado a lado e exportação das figuras em formato JPG.

# ## Considerações
#
# Este notebook possui finalidade apenas de estudo e de experimentação. As nuvens de palavras representam frequências de termos e não constituem, isoladamente, evidência de capacidade discriminativa ou desempenho de classificação.

# ## Referências
# [1] Wang, William Yang. "“Liar, liar pants on fire”: A new benchmark dataset for fake news detection." Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). 2017.
#
# [2] [LIAR — Hugging Face](https://huggingface.co/datasets/ucsbai/liar)
#
# [3] Upadhayay, Bibek, and Vahid Behzadan. "Sentimental liar: Extended corpus and deep learning models for fake claim classification." 2020 IEEE International Conference on Intelligence and Security Informatics (ISI). IEEE, 2020.
#
# [4] Alhindi, Tariq, Savvas Petridis, and Smaranda Muresan. "Where is your evidence: Improving fact-checking by justification modeling." Proceedings of the first workshop on fact extraction and verification (FEVER). 2018.
#
# [5] Shushkevich, Elena, John Cardiff, and Anna Boldyreva. "Detection of truthful, semi-truthful, false and other news with arbitrary topics using BERT-based models." 2023 33rd Conference of Open Innovations Association (FRUCT). IEEE, 2023.

# # Implementação

# ============================================================
# 1. INSTALAÇÃO E IMPORTAÇÕES
# ============================================================



import re
from collections import Counter
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import requests

from nltk.corpus import stopwords
from wordcloud import WordCloud
nltk.download("stopwords", quiet=True)
import io
import zipfile
import requests
from pathlib import Path

SEED = 42
STOPWORDS = set(stopwords.words("english"))

DATASET_URL = "https://datasets-server.huggingface.co"
DATASET_NAME = "ucsbai/liar"
LIAR_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"

# ============================================================
# 2. DEFINIÇÃO DAS CLASSES
# ============================================================

LIAR_CLASSES = {"pants-fire": 0,"false": 1,"barely-true": 2,"half-true": 3, "mostly-true": 4,"true": 5,}
SENTIMENTAL_CLASSES = {"pants-fire": 0, "false": 0, "barely-true": 0, "half-true": 0, "mostly-true": 1, "true": 1,}
BINARY_CLASSES = {"pants-fire": 0, "false": 0, "barely-true": 0, "half-true": 1, "mostly-true": 1, "true": 1,}
TERNARY_CLASSES = {"pants-fire": 0, "false": 0, "barely-true": 1, "half-true": 1, "mostly-true": 1,"true": 2,}

CLASS_NAMES = {
    "liar": {0: "Pants-fire", 1: "False", 2: "Barely-true", 3: "Half-true", 4: "Mostly-true", 5: "True",},
    "sentimental_liar": { 0: "False", 1: "True",},
    "binary_liar": { 0: "False", 1: "True",},
    "ternary_liar": { 0: "False", 1: "Mixed / Partially True", 2: "True",},
}

DATASET_MAPS = {
    "liar": LIAR_CLASSES,
    "sentimental_liar": SENTIMENTAL_CLASSES,
    "binary_liar": BINARY_CLASSES,
    "ternary_liar": TERNARY_CLASSES,
}

# Cores diferentes para cada classe de cada dataset.
CLASS_COLORS = [
    "#B71C1C",  # Vermelho
    "#EF6C00",  # Laranja
    "#F9A825",  # Amarelo
    "#00897B",  # Verde
    "#1565C0",  # Azul
    "#6A1B9A",  # Roxo
]

# ============================================================
# 3. CARREGAMENTO DO LIAR — ARQUIVOS ORIGINAIS
# ============================================================

def load_liar():
    response = requests.get(LIAR_URL, timeout=120)
    response.raise_for_status()
    files = {"train": "train.tsv", "validation": "valid.tsv", "test": "test.tsv",}
    splits = {}

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for split, filename in files.items():
            path = next(
                name for name in archive.namelist()
                if name.endswith(filename)
            )
            with archive.open(path) as file:
                df = pd.read_csv(
                    file,
                    sep="\t",
                    header=None,
                    dtype=str,
                )
            # Coluna 1: rótulo; coluna 2: afirmação.
            df = df.iloc[:, [2, 1]].copy()
            df.columns = ["statement", "original_label"]
            df["original_label"] = (
                df["original_label"]
                .str.strip()
                .str.lower()
            )
            splits[split] = df.reset_index(drop=True)
    return splits

splits = load_liar()
print("LIAR carregado com sucesso!")
for split, df in splits.items():
    print(f"{split}: {len(df)} registros")

# ============================================================
# 4. TRANSFORMAÇÃO DOS QUATRO DATASETS
# ============================================================

def create_label_columns(df):
    """Adiciona os quatro rótulos numéricos ao mesmo texto."""
    result = df.copy()
    # Normaliza nomes, sem alterar os textos originais.
    names = (
        result["original_label"]
        .str.lower()
        .str.strip()
        .str.replace("_", "-", regex=False)
    )
    for dataset, mapping in DATASET_MAPS.items():
        result[dataset] = names.map(mapping)
        if result[dataset].isna().any():
            unknown = names[result[dataset].isna()].unique()
            raise ValueError(
                f"Rótulos não reconhecidos em {dataset}: {unknown}"
            )
        result[dataset] = result[dataset].astype(int)
    return result

splits = {
    name: create_label_columns(df)
    for name, df in splits.items()
}

# DataFrame completo para análise exploratória.
df_liar = pd.concat(
    splits.values(),
    ignore_index=True,
)
columns = ["statement", "original_label", "liar", "sentimental_liar", "binary_liar", "ternary_liar",]

classes = ["true", "pants-fire", "half-true"]
max_words = 8
indices = [2411, 7653, 392] #textos curtos de exemplos
print("\nExemplo de textos e classes transformadas:")
print(df_liar.loc[indices, columns].to_string(index=True))
#df_short = df_liar[df_liar["original_label"].isin(classes) & (df_liar["statement"].fillna("").str.split().str.len() <= max_words)]
#print(f"\nExemplo de textos com até {max_words} palavras por classe:")
#display(df_short.groupby("original_label", group_keys=False).sample(n=1, random_state=42)[columns])

# ============================================================
# 5. DISTRIBUIÇÃO E BALANCEAMENTO DAS CLASSES
# ============================================================
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

def plot_class_distribution_all(df):
    """Plota os quatro datasets lado a lado e salva em JPG."""
    datasets = list(DATASET_MAPS.keys())
    fig, axes = plt.subplots(1, len(datasets), figsize=(24, 5.5))
    for ax, dataset in zip(axes, datasets):
        class_names = CLASS_NAMES[dataset]
        counts = df[dataset].value_counts().reindex(class_names.keys(), fill_value=0)
        total = counts.sum()
        percentages = counts / total * 100
        labels = [
            f"{class_id}\n{class_name}"
            for class_id, class_name in class_names.items()
        ]
        colors = [
            CLASS_COLORS[class_id]
            for class_id in class_names
        ]
        bars = ax.bar(range(len(labels)), counts.values, color=colors, width=0.7, edgecolor="white")
        # Quantidade e percentual acima das barras
        for bar, count, pct in zip(bars, counts.values, percentages.values):
            ax.annotate(
                f"{count:,}\n({pct:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9
            )
        ax.set_title(
            dataset.replace("_", " ").title(),
            fontsize=14,
            fontweight="bold"
        )
        ax.set_xlabel("Classes")
        ax.set_ylabel("Amostras")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, max(counts.values) * 1.20)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Distribuição e balanceamento das classes",
        fontsize=17,
        fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # Salvar todos os gráficos em um único JPG
    filename = OUTPUT_DIR / "distribuicao_classes_todos_datasets.jpg"
    fig.savefig( filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)
    print(f"Imagem salva: {filename}")

# Gerar figura
plot_class_distribution_all(df_liar)

# ============================================================
# 6. PRÉ-PROCESSAMENTO PARA NUVENS DE PALAVRAS
# Pré-processamento: tratamento de valores nulos, conversão para minúsculas, remoção de URLs, números, pontuação,
#caracteres especiais, stopwords e palavras com menos de 3 caracteres, seguido de tokenização e reconstrução do texto.
# ============================================================

def clean_text(text):
    """Normaliza o texto para visualização das palavras."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [
        word
        for word in text.split()
        if len(word) > 2 and word not in STOPWORDS
    ]
    return " ".join(words)
df_liar["clean_text"] = df_liar["statement"].map(clean_text)

# ============================================================
# 7. NUVENS DE PALAVRAS POR CLASSE
# ============================================================

def plot_wordclouds(df, dataset, max_words=100):
    """
    Gera uma nuvem para cada classe do dataset.
    Cada classe possui cor única.
    """
    class_names = CLASS_NAMES[dataset]
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(4.5 * n_classes, 4.5), constrained_layout=True,)
    # Garante comportamento uniforme com 2, 3 ou 6 classes.
    for ax, (class_id, class_name) in zip(axes, class_names.items()):
        texts = df.loc[df[dataset] == class_id,"clean_text",]
        frequencies = Counter(
            word
            for text in texts
            for word in text.split()
        )
        color = CLASS_COLORS[class_id]

        if frequencies:
            cloud = WordCloud(
                width=700,
                height=500,
                background_color="white",
                max_words=max_words,
                random_state=SEED,
                collocations=False,
                color_func=lambda *args, c=color, **kwargs: c,
            ).generate_from_frequencies(frequencies)
            ax.imshow(cloud, interpolation="bilinear")
        else:
            ax.text(
                0.5, 0.5, "Sem palavras",
                ha="center", va="center",
            )
        ax.set_title(
            f"{class_id} — {class_name}\n"
            f"({len(texts):,} textos)",
            fontsize=12,
            fontweight="bold",
            color=color,
        )
        ax.axis("off")

    fig.suptitle(
        f"Word Clouds — {dataset}",
        fontsize=16,
        fontweight="bold",
    )

    # Salvar as nuvens de cada dataset em um único JPG
    OUTPUT_DIR = Path("figures")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = OUTPUT_DIR / f"wordcloud_{dataset}.jpg"
    fig.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white",)
    plt.show()
    plt.close(fig)
    print(f"Imagem salva: {filename}")

# ============================================================
# 8. EXECUÇÃO PARA OS QUATRO DATASETS
# ============================================================

for dataset in DATASET_MAPS:
    plot_wordclouds(df_liar, dataset)

