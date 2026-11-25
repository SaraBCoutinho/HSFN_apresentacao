"""Experimentos didáticos de classificação e múltiplos classificadores com Iris.

Código adaptado de um notebook, com auxílio de Inteligência Artificial.
"""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use('Agg')  # Permite gerar imagens sem interface gráfica.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


PASTA_PADRAO = Path(__file__).resolve().parent / 'resultados'


def salvar(fig, pasta, nome):
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / nome
    fig.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Gráfico salvo: {caminho}')


def exemplos_knn():
    """Classifica uma amostra de teste de cada espécie com KNN."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    modelo = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
    print('\n--- Exemplos de classificação KNN (3 classes) ---')
    for classe in np.unique(y_test):
        indice = np.flatnonzero(y_test == classe)[0]
        amostra = X_test[indice:indice + 1]
        previsto = modelo.predict(amostra)[0]
        print(f'{iris.target_names[classe]} | atributos: {amostra[0]} | '
              f'predição: {iris.target_names[previsto]}')
    return pd.DataFrame(iris.data, columns=iris.feature_names).assign(
        especie=iris.target_names[iris.target]
    )


def grade(X, passo=0.02):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    return np.meshgrid(np.arange(x_min, x_max, passo),
                       np.arange(y_min, y_max, passo))


def classificacoes_e_oracle(pasta):
    # 1. Carregar o dataset Iris (usando 2 features para visualização 2D clara)
    iris = load_iris()
    X = iris.data[:, [2, 3]]  # Comprimento da pétala e Largura da pétala
    y = iris.target
    feature_names = ['Comprimento da Pétala (cm)', 'Largura da Pétala (cm)']
    target_names = iris.target_names

    # Divisão treino e teste (estratificada para manter proporção das classes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, random_state=42, stratify=y
    )

    # 2. Definir 3 classificadores com vieses indutivos diferentes
    classifiers = {
        "Árvore de Decisão (Max Depth=2)": DecisionTreeClassifier(max_depth=2, random_state=42),
        "SVM (Kernel Linear)": SVC(kernel="linear", C=1.0),
        "KNN (k=3 Vizinhos)": KNeighborsClassifier(n_neighbors=3)
    }

    # 3. Preparar a grade (meshgrid) para desenhar as regiões no espaço 2D
    h = 0.02  # passo da grade
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Paleta de cores suave para slides (Azul, Laranja, Verde)
    cmap_light = ListedColormap(['#AEC6CF', '#FFB347', '#B2E2B2'])
    cmap_bold = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Configurar figura em alta resolução (16:9 widescreen, ideal para Google Slides)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=300)
    plt.subplots_adjust(wspace=0.25)

    predictions = {}
    correct_masks = {}

    # 4. Treinar, avaliar e plotar as regiões de cada classificador
    for idx, (name, clf) in enumerate(classifiers.items()):
        ax = axes[idx]
        clf.fit(X_train, y_train)

        # Predição no conjunto de teste
        y_pred = clf.predict(X_test)
        predictions[name] = y_pred
        correct = (y_pred == y_test)
        correct_masks[name] = correct
        acc = np.mean(correct) * 100

        # Predição na grade para pintar a região de decisão
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

        # Plotar instâncias de teste: Acertos (círculos) vs Erros (X vermelho)
        for class_idx in range(3):
            idx_class = (y_test == class_idx)
            # Acertos da classe
            ax.scatter(
                X_test[idx_class & correct, 0], X_test[idx_class & correct, 1],
                c=cmap_bold[class_idx], label=target_names[class_idx] if idx == 0 else "",
                edgecolors='k', s=60, marker='o'
            )
            # Erros do classificador (destacados com X)
            ax.scatter(
                X_test[idx_class & ~correct, 0], X_test[idx_class & ~correct, 1],
                c='red', edgecolors='k', s=110, marker='X', linewidths=1.5,
                label='Erro do Modelo' if (idx == 0 and class_idx == 0) else ""
            )

        ax.set_title(f"{name}\nAcurácia Teste: {acc:.1f}%", fontsize=12, fontweight='bold')
        ax.set_xlabel(feature_names[0], fontsize=10)
        if idx == 0:
            ax.set_ylabel(feature_names[1], fontsize=10)

    # 5. Quarto Gráfico: Diversidade e Complementaridade no Espaço (Oráculo vs Consenso)
    ax_div = axes[3]
    names_list = list(classifiers.keys())
    c1, c2, c3 = correct_masks[names_list[0]], correct_masks[names_list[1]], correct_masks[names_list[2]]

    all_correct = c1 & c2 & c3
    oracle_correct = c1 | c2 | c3
    only_some = oracle_correct & ~all_correct  # Instâncias onde os classificadores discordam (uns acertam, outros erram!)
    all_wrong = ~oracle_correct

    # Plotar consenso vs regiões de discordância e complementaridade
    ax_div.scatter(X_test[all_correct, 0], X_test[all_correct, 1], c='#95a5a6', alpha=0.5, s=50, label='Todos Acertam (Consenso)')
    ax_div.scatter(X_test[only_some, 0], X_test[only_some, 1], c='#8e44ad', edgecolors='k', s=130, marker='D', label='Acertos Complementares (Diversidade)')
    ax_div.scatter(X_test[all_wrong, 0], X_test[all_wrong, 1], c='red', edgecolors='k', s=110, marker='X', label='Todos Erram')

    oracle_acc = np.mean(oracle_correct) * 100
    ax_div.set_title(f"Complementaridade (MCS)\nAcurácia Oráculo: {oracle_acc:.1f}%", fontsize=12, fontweight='bold', color='#5b2c6f')
    ax_div.set_xlabel(feature_names[0], fontsize=10)
    ax_div.legend(loc='upper left', fontsize=8.5, framealpha=0.9)

    # Legenda geral no primeiro gráfico
    axes[0].legend(loc='upper left', fontsize=8.5, framealpha=0.9)

    plt.suptitle("Classificações e Oracle (Dataset Iris)", fontsize=15, fontweight='bold', y=1.04)

    # Salvar imagem transparente/alta resolução pronta para colar no Google Slides
    salvar(fig, pasta, "classificacoes_e_oracle.png")

    # 6. Resumo textual no terminal mostrando exatamente as instâncias onde eles divergem
    print("\n--- ANÁLISE DE DIVERSIDADE NAS INSTÂNCIAS DE TESTE ---")
    for i in range(len(y_test)):
        acertos = [nome.split()[0] for nome in names_list if correct_masks[nome][i]]
        if 0 < len(acertos) < 3:
            print(f"Instância #{i:02d} (Classe real: {target_names[y_test[i]]}) -> Acertada APENAS por: {', '.join(acertos)}")


def main():
    exemplos_knn()
    classificacoes_e_oracle(PASTA_PADRAO)


if __name__ == "__main__":
    main()
