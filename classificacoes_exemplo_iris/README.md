# Classificações e Oracle — Iris

Experimento educacional com o dataset Iris. Compara árvore de decisão (profundidade 2), SVM com kernel linear (C=1.0) e KNN (k=3), utilizando comprimento e largura da pétala e divisão estratificada de treino e teste (35% para teste).

O gráfico apresenta as fronteiras de decisão, acurácias de teste, erros e complementaridade dos três classificadores. O quarto painel apresenta o oráculo, limite superior hipotético que considera uma instância correta quando pelo menos um modelo acerta. Não representa um seletor implementado.

## Execução

```bash
pip install -r requirements.txt
python classificacao.py
```

O script também apresenta exemplos de KNN e salva a figura `resultados/classificacoes_e_oracle.png` com quatro painéis, em 300 dpi.

## Finalidade e uso de IA

O código foi desenvolvido e adaptado com auxílio de Inteligência Artificial. Este projeto possui finalidade exclusivamente educacional, de estudo e experimentação, sem pretensão de uso em produção.
