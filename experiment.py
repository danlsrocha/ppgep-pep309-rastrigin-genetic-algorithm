#!/bin/sh
''''exec "$(d=$(dirname "$0"); [ -x "$d/.venv/bin/python" ] && echo "$d/.venv/bin/python" || echo python3)" "$0" "$@" # '''
"""Experimento final: 30 execuções aleatorizadas de cada caso de teste, estatísticas, teste T pareado e gráficos.

Uso:  python experiment.py [--runs 30] [--seed 2026]
Saídas (pasta results/): runs.csv, summary.json, convergence.png, boxplot.png
"""

# -------------------------------------------------------------------- #
# Importações
# -------------------------------------------------------------------- #
import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Backend sem janela (apenas grava arquivos)
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import ga


# -------------------------------------------------------------------- #
# Experimento
# -------------------------------------------------------------------- #
def main() -> None:
    """Executa o experimento completo: roda o AG nos casos de teste,
     grava runs.csv e summary.json, e gera os gráficos de convergência
     e de boxplot.
    """
    # ---------------------------------------------------------------- #
    # Argumentos de linha de comando
    # ---------------------------------------------------------------- #
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026,
                    help="semente mestra (reprodutibilidade)")
    args = ap.parse_args()

    cases = ga.default_cases()

    # Uma semente independente por execução, gerada a partir da semente
    # mestra. A execução i de C1 e a execução i de C2 partem da MESMA
    # população inicial (é isso que forma os pares do teste T pareado);
    # os fluxos aleatórios da evolução são independentes.
    run_seeds = np.random.SeedSequence(args.seed).generate_state(args.runs)

    # Estruturas para armazenar os resultados de cada caso
    finals = {k: [] for k in cases}      # Custo final de cada execução
    times = {k: [] for k in cases}       # Tempo de cada execução (s)
    histories = {k: [] for k in cases}   # Histórico de mínimos por geração
    best_x = {}                          # Melhor (custo, solução) por caso

    # ---------------------------------------------------------------- #
    # Execuções do AG
    # ---------------------------------------------------------------- #
    # Embaralha a ordem de execução dos pares (caso, execução)
    order = [(k, i) for k in cases for i in range(args.runs)]
    np.random.default_rng(args.seed).shuffle(order)

    results = {}

    for k, i in order:
        cfg = cases[k]

        # Mesma população inicial para a execução i de todos os casos
        pop = ga.init_population(cfg.pop_size, int(run_seeds[i]))

        # Fluxo aleatório da evolução, independente entre os casos
        rng = np.random.default_rng([int(run_seeds[i]), int(k[1:])])

        t0 = time.perf_counter()
        x, cost, hist = ga.run_ga(cfg, rng, pop)

        results[(k, i)] = (cost, time.perf_counter() - t0, hist, x)
        print(f"{k} execução {i:2d}: {cost:.6f}")

    # Reorganiza os resultados por caso, na ordem das execuções
    for k in cases:
        for i in range(args.runs):
            cost, dt, hist, x = results[(k, i)]

            finals[k].append(cost)
            times[k].append(dt)
            histories[k].append(hist)

            # Guarda a melhor solução encontrada em todas as execuções
            if k not in best_x or cost < best_x[k][0]:
                best_x[k] = (cost, x)

    # ---------------------------------------------------------------- #
    # Resultados brutos (runs.csv)
    # ---------------------------------------------------------------- #
    out = Path("results")
    out.mkdir(exist_ok=True)

    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "seed"]
                   + [f"{k}_final" for k in cases]
                   + [f"{k}_time_s" for k in cases])

        for i in range(args.runs):
            w.writerow([i, int(run_seeds[i])]
                       + [f"{finals[k][i]:.10f}" for k in cases]
                       + [f"{times[k][i]:.4f}" for k in cases])

    # ---------------------------------------------------------------- #
    # Estatísticas descritivas por caso
    # ---------------------------------------------------------------- #
    summary = {"master_seed": args.seed, "runs": args.runs, "cases": {}}

    for k, cfg in cases.items():
        a = np.array(finals[k])

        summary["cases"][k] = {
            "name": cfg.name,
            "best": a.min(),
            "worst": a.max(),
            "mean": a.mean(),
            "std": a.std(ddof=1),
            "median": float(np.median(a)),
            "mean_time_s": float(np.mean(times[k])),
            "best_solution_max_abs_x": float(np.abs(best_x[k][1]).max()),
        }

    # ---------------------------------------------------------------- #
    # Teste estatístico pareado (C1 vs. C2)
    # ---------------------------------------------------------------- #
    c1, c2 = np.array(finals["C1"]), np.array(finals["C2"])
    t, p = stats.ttest_rel(c1, c2)
    diff = c1 - c2

    summary["paired_t_test"] = {
        "t": float(t),
        "p_value": float(p),
        "df": args.runs - 1,
        "mean_difference_C1_minus_C2": float(diff.mean()),
        # Normalidade das diferenças (pressuposto do teste T) e
        # alternativa não paramétrica (Wilcoxon)
        "shapiro_p_differences": float(stats.shapiro(diff).pvalue),
        "wilcoxon_p": float(stats.wilcoxon(c1, c2).pvalue),
        "reject_H0_at_0.05": bool(p < 0.05),
    }

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(json.dumps(summary, indent=2, default=float))

    # ---------------------------------------------------------------- #
    # Gráfico de convergência (média ± desvio padrão por geração)
    # ---------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for k in cases:
        h = np.array(histories[k])
        m, s = h.mean(0), h.std(0, ddof=1)

        ax.plot(m, label=cases[k].name)
        ax.fill_between(np.arange(len(m)), m - s, m + s, alpha=0.2)

    ax.set_yscale("log")
    ax.set_xlabel("Geração")
    ax.set_ylabel("Melhor valor de Rastrigin (escala log)")
    ax.set_title(
        f"Convergência (média ± desvio padrão em {args.runs} execuções)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=150)

    # ---------------------------------------------------------------- #
    # Boxplot dos resultados finais
    # ---------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(5, 4.5))

    ax.boxplot([c1, c2], tick_labels=["C1", "C2"])
    ax.set_yscale("log")
    ax.set_ylabel("Melhor valor final (escala log)")
    ax.set_title("Resultados finais nas execuções")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "boxplot.png", dpi=150)


if __name__ == "__main__":
    main()
