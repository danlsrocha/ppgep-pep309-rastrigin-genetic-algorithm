#!/bin/sh
''''exec "$(d=$(dirname "$0"); [ -x "$d/.venv/bin/python" ] && echo "$d/.venv/bin/python" || echo python3)" "$0" "$@" # '''
"""Busca em grade (grid search) piloto utilizada para escolher os parâmetros do AG em cada caso de teste.

Usa sementes disjuntas das do experimento final (experiment.py) e o mesmo
orçamento para os dois casos, de modo que nenhum seja favorecido.
Grava results/tuning.csv e imprime a melhor configuração de cada caso.
"""

# -------------------------------------------------------------------- #
# Importações
# -------------------------------------------------------------------- #
import csv
import itertools
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

import ga


# -------------------------------------------------------------------- #
# Constantes da Busca
# -------------------------------------------------------------------- #
PILOT_RUNS = 10             # Execuções piloto por configuração
PILOT_SEED_BASE = 1_000_000 # O experimento final usa outra faixa de sementes

# Parâmetros comuns aos dois casos
POP_SIZES = [400, 800]
PCS = [0.9, 1.0]

# Grades por caso sobre (probabilidade de mutação, parâmetro da
# mutação, parâmetro do crossover).
# Rodada 2: refinamento em torno da melhor região da rodada 1 (a rodada
# 1 está em results/tuning_round1.csv).
GRIDS = {
    "C1": {
        "pm": [0.02, 0.05, 0.1],
        "mut": ("sigma", [0.01, 0.02, 0.05]),
        "xover": ("alpha", [0.0, 0.05, 0.1, 0.2]),
    },
    "C2": {
        "pm": [0.2, 0.3, 0.5],
        "mut": ("b", [4.0, 5.0, 7.0]),
        "xover": None,
    },
}


# -------------------------------------------------------------------- #
# Avaliação de uma Configuração
# -------------------------------------------------------------------- #
def evaluate(
    args: tuple[str, ga.Config]
) -> tuple[str, ga.Config, float, float]:
    """Executa PILOT_RUNS vezes o AG com uma configuração e resume o
     custo final obtido.

    Args:
        args (tuple[str, ga.Config]): Chave do caso ("C1" ou "C2") e
         configuração a ser avaliada (uma tupla única, para ser usada
         com ProcessPoolExecutor.map).

    Returns:
        tuple[str, ga.Config, float, float]: Chave do caso, configuração
         avaliada, média e desvio padrão amostral (ddof=1) dos custos
         finais.
    """
    key, cfg = args
    finals = []

    for i in range(PILOT_RUNS):
        # A mesma semente define a população inicial e o gerador do AG
        seed = PILOT_SEED_BASE + i
        pop = ga.init_population(cfg.pop_size, seed)
        rng = np.random.default_rng([seed, 1])

        # run_ga retorna (melhor indivíduo, custo, histórico)
        finals.append(ga.run_ga(cfg, rng, pop)[1])

    return key, cfg, float(np.mean(finals)), float(np.std(finals, ddof=1))


# -------------------------------------------------------------------- #
# Busca em Grade
# -------------------------------------------------------------------- #
def main() -> None:
    """Monta a grade de cada caso, avalia todas as configurações em
     paralelo, grava results/tuning.csv e imprime as 5 melhores
     configurações de cada caso.
    """
    # Monta a lista de trabalhos (caso, configuração)
    jobs = []

    for key, base in ga.default_cases().items():
        g = GRIDS[key]
        mut_name, mut_values = g["mut"]
        x_name, x_values = g["xover"] or ("-", [None])

        # Produto cartesiano de todos os parâmetros da grade
        combinations = itertools.product(
            POP_SIZES, PCS, g["pm"], mut_values, x_values)

        for pop, pc, pm, mv, xv in combinations:
            xp = dict(base.xover_params)
            if x_name != "-":
                xp[x_name] = xv

            cfg = replace(base, pop_size=pop, pc=pc, pm=pm,
                          xover_params=xp,
                          mut_params={**base.mut_params, mut_name: mv})
            jobs.append((key, cfg))

    # Avalia as configurações em paralelo (um processo por núcleo)
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(evaluate, jobs, chunksize=4))

    # Grava todos os resultados em results/tuning.csv
    out = Path("results")
    out.mkdir(exist_ok=True)

    with open(out / "tuning.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "pop_size", "pc", "pm", "xover_params",
                    "mut_params", "mean_final", "std_final"])

        for key, cfg, mean, std in results:
            w.writerow([key, cfg.pop_size, cfg.pc, cfg.pm,
                        cfg.xover_params, cfg.mut_params,
                        f"{mean:.4f}", f"{std:.4f}"])

    # Imprime as 5 melhores configurações de cada caso (menor custo médio)
    for key in GRIDS:
        subset = sorted((r for r in results if r[0] == key),
                        key=lambda r: r[2])

        print(f"\n{key}: 5 melhores de {len(subset)} configurações "
              f"(custo final médio em {PILOT_RUNS} execuções)")

        for _, cfg, mean, std in subset[:5]:
            print(f"  pop={cfg.pop_size} pc={cfg.pc} pm={cfg.pm} "
                  f"{cfg.xover_params} {cfg.mut_params} "
                  f"-> {mean:.3f} +- {std:.3f}")


if __name__ == "__main__":
    main()
