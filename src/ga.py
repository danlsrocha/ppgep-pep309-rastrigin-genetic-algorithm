"""Algoritmo Genético para a Função de Rastrigin.

Comum a todos os casos de teste:
    * representação real
    * inicialização aleatória uniforme em [-5,12; 5,12]^n
    * fitness via ranking linear
    * seleção por roleta SUS (stochastic universal sampling)
    * elitismo
    * parada após um número fixo de gerações

Os casos de teste diferem nos operadores de crossover e de mutação (ver
 default_cases).
"""

# -------------------------------------------------------------------- #
# Importações
# -------------------------------------------------------------------- #
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np


# -------------------------------------------------------------------- #
# Constantes do Problema
# -------------------------------------------------------------------- #
A = 10.0                # Amplitude da função de Rastrigin (constante)
LB = -5.12              # Limite inferior (Lower Bound)
UB = 5.12               # Limite superior (Upper Bound)
N_VARS = 100            # Número de variáveis
N_GENERATIONS = 300     # Número de gerações


# -------------------------------------------------------------------- #
# Gerador de População Inicial
# -------------------------------------------------------------------- #
def init_population(pop_size: int, seed: int) -> np.ndarray:
    """Gera uma população inicial (matriz) de indivíduos uniformemente
     distribuídos no espaço de busca definido pelos limites LB e UB.
    
    Args:
        pop_size (int): Tamanho da população.
        seed (int): Semente do gerador de números aleatórios (permite
         reproduzibilidade).

    Returns:
        np.ndarray: Matriz de população inicial, no formato
         (pop_size, n_vars).
    """
    return np.random.default_rng(seed).uniform(LB, UB, (pop_size, N_VARS))


# -------------------------------------------------------------------- #
# Cálculo de Custos via Função de Rastrigin
# -------------------------------------------------------------------- #
def rastrigin(
    pop: np.ndarray
) -> np.ndarray:
    """Calcula um vetor de custos via função de Rastrigin a partir de
     uma matriz de população (ou vetor de indivíduo).

    Args:
        pop (np.ndarray): Matriz de população, no formato 
         (n_individuos,n_vars) ou (n_vars,).
    
    Returns:
        np.ndarray: Vetor de custos (valores da função de Rastrigin), no 
         formato (n_individuos,) ou (1,).
    """
    # Garante que pop seja vista como bidimensional
    pop = np.atleast_2d(pop)

    # Número de variáveis (colunas)
    n = pop.shape[1] 

    # Retorna um vetor de custos (valores da função de Rastringin)
    return A * n + np.sum(pop**2 - A * np.cos(2.0 * np.pi * pop), axis=1)


# -------------------------------------------------------------------- #
# Cálculo de Fitness via LRS (linear ranking selection)
# -------------------------------------------------------------------- #
def linear_ranking_fitness(
    costs: np.ndarray, 
    s: float
) -> np.ndarray:
    """Calcula o fitness de uma população via seleção por ranking
     linear a partir de um vetor de custos.

    Args:
        costs (np.ndarray): Vetor de custos da população, no formato 
         ([n_individuos],).
        s (float): Pressão seletiva, no intervalo [1, 2]. 
    
    Returns:
        np.ndarray: Vetor de fitness, no formato ([n_individuos],).
    """
    # Número de indivíduos na população
    n = len(costs) 

    # Ordena os custos em ordem decrescente (+custo = -fitness)
    order = np.argsort(-costs, kind="stable")

    # Cria um vetor de ranks, (melhor indivíduo = 0; pior = n-1)
    ranks = np.empty(n) 
    ranks[order] = np.arange(n)

    # Retorna um vetor de fitness
    return (2.0 - s) + 2.0 * (s - 1.0) * ranks / (n - 1)


# -------------------------------------------------------------------- #
# Seleção via Roleta SUS (stochastic universal sampling)
# -------------------------------------------------------------------- #
def sus_selection(
    fitness: np.ndarray, 
    n_select: int, 
    rng: np.random.Generator
) -> np.ndarray:
    """Seleciona n_select indivíduos via roleta SUS (Stochastic
     Universal Sampling) a partir de um vetor de fitness.

    Args:
        fitness (np.ndarray): Vetor de fitness da população, no formato
         ([n_individuos],).
        n_select (int): Número de indivíduos a serem selecionados.
        rng (np.random.Generator): Gerador de números aleatórios.

    Returns:
        np.ndarray: Vetor de índices dos indivíduos selecionados, no
            formato ([n_select],).
    """
    # Vetor de probabilidades cumulativas (cumsum) normalizado para 1
    cumulative: np.ndarray = np.cumsum(fitness / fitness.sum())
    cumulative[-1] = 1.0 # proteção contra erro de arredondamento

    # Gera n_select ponteiros igualmente espaçados no intervalo [0, 1)
    pointers: np.ndarray = (rng.random() + np.arange(n_select)) / n_select

    # Retorna os índices dos indivíduos selecionados (um por ponteiro)
    return np.searchsorted(cumulative, pointers)


# -------------------------------------------------------------------- #
# Crossover via BLX-alpha (Blend Crossover)
# -------------------------------------------------------------------- #
def blx_alpha(
    p1: np.ndarray,
    p2: np.ndarray,
    rng: np.random.Generator,
    alpha: float = 0.5,
    **_,
) -> tuple[np.ndarray, np.ndarray]:
    """Gera dois filhos a partir de dois pais via BLX-alpha (Blend
     Crossover).

    Args:
        p1 (np.ndarray): Primeiros pais de cada par, no formato
         (n_pares, n_vars) ou (n_vars,).
        p2 (np.ndarray): Segundos pais de cada par, no formato
         (n_pares, n_vars) ou (n_vars,).
        rng (np.random.Generator): Gerador de números aleatórios.
        alpha (float): Balanço entre exploitação e exploração (0.0 = 
         apenas exploitação, 1.0 = apenas exploração).

    Returns:
        tuple[np.ndarray, np.ndarray]: Dois filhos gerados a partir dos
         pais, cada um com o mesmo formato de p1 e p2.
    """
    # Define os limites inferior e superior do intervalo de exploração
    lo: np.ndarray = np.minimum(p1, p2)
    hi: np.ndarray = np.maximum(p1, p2)

    # Calcula as distâncias entre os pais
    d: np.ndarray = hi - lo

    # Expande o intervalo de exploração em alpha * d para cada lado
    lo: np.ndarray = lo - alpha * d
    hi: np.ndarray = hi + alpha * d

    # Gera dois filhos uniformemente distribuídos no intervalo expandido
    c1: np.ndarray = rng.uniform(lo, hi)
    c2: np.ndarray = rng.uniform(lo, hi)
    
    # Retorna os dois filhos gerados
    return c1, c2


# -------------------------------------------------------------------- #
# Crossover Aritmético
# -------------------------------------------------------------------- #
def arithmetic(
    p1: np.ndarray, 
    p2: np.ndarray, 
    rng: np.random.Generator, 
    **_
) -> tuple[np.ndarray, np.ndarray]:
    """Gera dois filhos a partir de dois pais via crossover aritmético
     inteiro contrativo com peso aleatório por gene.
    
    Args:
        p1 (np.ndarray): Primeiros pais de cada par, no formato
         (n_pares, n_vars) ou (n_vars,).
        p2 (np.ndarray): Segundos pais de cada par, no formato
         (n_pares, n_vars) ou (n_vars,).
        rng (np.random.Generator): Gerador de números aleatórios.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Dois filhos gerados a partir dos
         pais, cada um com o mesmo formato de p1 e p2.
    """
    # Vetor de lambdas aleatórios no intervalo [0, 1)
    lam: np.ndarray = rng.random(p1.shape)

    # Retorna os dois filhos gerados
    return lam * p1 + (1 - lam) * p2, lam * p2 + (1 - lam) * p1


# -------------------------------------------------------------------- #
# Mutação Gaussiana
# -------------------------------------------------------------------- #
def gaussian_mutation(
    x: np.ndarray, 
    rng: np.random.Generator, 
    pm: float, 
    sigma: float, 
    **_
) -> np.ndarray:
    """Aplica mutação gaussiana aos indivíduos.
    
    Args:
        x (np.ndarray): Descendentes a serem mutados, no formato
         (n_individuos, n_vars) ou (n_vars,).
        rng (np.random.Generator): Gerador de números aleatórios.
        pm (float): Probabilidade de mutação por gene.
        sigma (float): Desvio padrão da distribuição gaussiana, como
         fração da amplitude do espaço de busca (UB - LB).

    Returns:
        np.ndarray: Descendentes mutados, com o mesmo formato de x.
    """
    mask: np.ndarray = rng.random(x.shape) < pm
    noise: np.ndarray = rng.normal(0.0, sigma * (UB - LB), x.shape)
    return x + mask * noise


# -------------------------------------------------------------------- #
# Mutação Não Uniforme
# -------------------------------------------------------------------- #
def non_uniform_mutation(
    x: np.ndarray, 
    rng: np.random.Generator, 
    pm: float, 
    gen: int, 
    max_gen: int, 
    b: float, 
    **_
) -> np.ndarray:
    """Aplica mutação não uniforme de Michalewicz aos indivíduos.

    Args:
        x (np.ndarray): Descendentes a serem mutados, no formato
         (n_individuos, n_vars) ou (n_vars,).
        rng (np.random.Generator): Gerador de números aleatórios.
        pm (float): Probabilidade de mutação por gene.
        gen (int): Número da geração atual (1 a max_gen).
        max_gen (int): Número máximo de gerações.
        b (float): Parâmetro de controle da não uniformidade da mutação
         (quanto maior, mais rápido a mutação diminui com o tempo).

    Returns:
        np.ndarray: Descendentes mutados, com o mesmo formato de x.
    """

    # Gera máscara de genes a serem mutados
    mask: np.ndarray = rng.random(x.shape) < pm

    # Calcula o fator de redução da mutação com o tempo
    shrink: np.ndarray = (
        1.0 - rng.random(x.shape) ** ((1.0 - gen / max_gen) ** b))

    # Gera máscara de genes a serem mutados para cima ou para baixo
    up: np.ndarray = rng.random(x.shape) < 0.5

    # Calcula o delta de mutação para cada gene (positivo ou negativo)
    delta: np.ndarray = np.where(up, (UB - x) * shrink, -(x - LB) * shrink)

    # Retorna o indivíduo mutado
    return x + mask * delta


# -------------------------------------------------------------------- #
# Definição de Parâmetros e Casos de Teste
# -------------------------------------------------------------------- #
class MutationOperator(Protocol):
    """Protocolo para operadores de mutação."""
    def __call__(
        self,
        x: np.ndarray,
        rng: np.random.Generator,
        pm: float,
        **kwargs: object,
    ) -> np.ndarray: ...


@dataclass
class Config:
    """Classe de configuração de um caso de teste do Algoritmo Genético.

    Attributes:
        name (str): Nome do caso de teste.
        crossover (Callable): Função de crossover utilizada.
        mutation (Callable): Função de mutação utilizada.
        pop_size (int): Tamanho da população.
        pc (float): Probabilidade de crossover (por par).
        pm (float): Probabilidade de mutação (por gene).
        s (float): Pressão seletiva do ranking linear.
        n_elite (int): Número de indivíduos elitistas.
        xover_params (dict[str, object]): Parâmetros do crossover.
        mut_params (dict[str, object]): Parâmetros da mutação.
    """
    name: str
    crossover: Callable # type: ignore
    mutation: MutationOperator
    pop_size: int = 100
    pc: float = 0.9
    pm: float = 0.01
    s: float = 1.8
    n_elite: int = 2
    xover_params: dict[str, object] = (
        field(default_factory=lambda: dict[str, object]()))
    mut_params: dict[str, object] = (
        field(default_factory=lambda: dict[str, object]()))


def default_cases() -> dict[str, Config]:
    """
    Define os casos de teste padrão (C1 e C2) com os operadores e
     parâmetros escolhidos na busca em grade piloto de tune.py.

    Args:
        None

    Returns:
        dict[str, Config]: Dicionário de casos de teste.
    """
    return {
        "C1": Config(
            "C1: BLX-alpha + Mutação Gaussiana", 
            blx_alpha,
            gaussian_mutation,
            pop_size=800, 
            pc=1.0, 
            pm=0.02,
            xover_params={"alpha": 0.1}, 
            mut_params={"sigma": 0.01}
        ),
        "C2": Config(
            "C2: Crossover Aritmético + Mutação Não Uniforme",
            arithmetic, 
            non_uniform_mutation,
            pop_size=800, 
            pc=1.0, 
            pm=0.5,
            mut_params={"b": 7.0}
        ),
    }


# -------------------------------------------------------------------- #
# Execução do Algoritmo Genético
# -------------------------------------------------------------------- #
def run_ga(
    cfg: Config, 
    rng: np.random.Generator, 
    pop: np.ndarray,
    n_generations: int = N_GENERATIONS
) -> tuple[np.ndarray, float, np.ndarray]:
    """ Executa o Algoritmo Genético para um caso de teste específico.

    Args:
        cfg (Config): Configuração do caso de teste.
        rng (np.random.Generator): Gerador de números aleatórios.
        pop (np.ndarray): População inicial, no formato
         (pop_size, n_vars).
        n_generations (int): Número de gerações a serem executadas.
    
    Returns:
        tuple[np.ndarray, float, np.ndarray]: Tupla contendo o melhor
         indivíduo encontrado, seu custo e o histórico de custos mínimos
         por geração, no formato (n_generations + 1,).
    """

    # Cria cópia da população inicial para não alterá-la fora da função
    pop = pop.copy()

    # Calcula os custos da população inicial
    costs = rastrigin(pop)

    # Cria vetor de histórico de custos mínimos por geração
    history = np.empty(n_generations + 1)
    
    # Armazena o custo mínimo da população inicial no histórico
    history[0] = costs.min()

    # Calcula o número de descendentes a serem gerados
    n_off = cfg.pop_size - cfg.n_elite

    # Garante que o número de descendentes (e de pais) seja par
    n_off += n_off % 2

    # Executa o loop principal do Algoritmo Genético por n_generations
    for gen in range(1, n_generations + 1):

        # Seleciona os indivíduos elitistas da população atual
        elite = pop[np.argsort(costs)[: cfg.n_elite]]

        # Calcula o fitness da população atual via ranking linear
        fitness = linear_ranking_fitness(costs, cfg.s)

        # Seleciona os pais via roleta SUS
        parents = pop[sus_selection(fitness, n_off, rng)]

        # Embaralha os pais (para evitar que a ordem afete o crossover)
        rng.shuffle(parents)

        # Divide os pais em pares (p1, p2) para o crossover
        p1, p2 = parents[0::2], parents[1::2]

        # Gera os descendentes possíveis via crossover
        c1, c2 = cfg.crossover(p1, p2, rng, **cfg.xover_params) # type: ignore

        # Decide que cruzamentos realmente ocorrerão
        do_x = (rng.random(len(p1)) < cfg.pc)[:, None]

        # Mantém os descendentes ou substitui pelos pais
        c1 = np.where(do_x, c1, p1) # type: ignore
        c2 = np.where(do_x, c2, p2) # type: ignore

        # Combina os descendentes em uma única matriz
        off: np.ndarray = np.vstack([c1, c2])[: cfg.pop_size - cfg.n_elite]

        # Aplica a mutação aos descendentes
        off = cfg.mutation(off, rng, pm=cfg.pm, gen=gen, 
                           max_gen=n_generations, **cfg.mut_params)

        # Restringe os genes dos descendentes aos limites [LB, UB]
        off = np.clip(off, LB, UB)

        # Combina descendentes e elitistas para formar a nova população
        pop = np.vstack([elite, off])

        # Calcula os custos da nova população
        costs = rastrigin(pop)

        # Armazena o custo mínimo da nova população no histórico
        history[gen] = costs.min()

    # Seleciona o melhor indivíduo da população final
    best = costs.argmin()

    # Retorna o melhor indivíduo, seu custo, e o histórico de mínimos
    return pop[best], costs[best], history