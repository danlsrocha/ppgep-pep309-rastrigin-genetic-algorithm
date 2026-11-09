"""Ponto de entrada multiplataforma (Windows, macOS, Linux): prepara o ambiente e executa os scripts.

Uso:
    python run.py experiment [--runs 30] [--seed 2026]  # experimento final
    python run.py tune                                  # busca em grade dos parâmetros
    python run.py setup                                 # apenas cria o .venv e instala as dependências

Na primeira execução cria o ambiente virtual `.venv` e instala o requirements.txt;
nas seguintes, reaproveita o ambiente (reinstala só se o requirements.txt mudar).
Usa apenas a biblioteca padrão, então basta ter o Python 3 instalado.
"""

# -------------------------------------------------------------------- #
# Importações
# -------------------------------------------------------------------- #
import subprocess
import sys
import venv
from pathlib import Path


# -------------------------------------------------------------------- #
# Constantes
# -------------------------------------------------------------------- #
RAIZ = Path(__file__).resolve().parent
VENV = RAIZ / ".venv"
REQUISITOS = RAIZ / "requirements.txt"

# Cópia do requirements.txt já instalado (detecta mudanças nas dependências)
MARCA = VENV / ".requirements.instalado"

# Comandos que executam um script do projeto
SCRIPTS = {"experiment": "experiment.py", "tune": "tune.py"}


# -------------------------------------------------------------------- #
# Ambiente Virtual
# -------------------------------------------------------------------- #
def python_do_venv() -> Path:
    """Retorna o caminho do executável do Python dentro do `.venv`.

    Returns:
        Path: Caminho do executável (Scripts/python.exe no Windows;
         bin/python nos demais sistemas).
    """
    return VENV / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def preparar_ambiente() -> Path:
    """Garante que o `.venv` exista e que as dependências do
     requirements.txt estejam instaladas nele.

    Cria o ambiente se necessário e reinstala as dependências apenas
    quando o requirements.txt mudou desde a última instalação.

    Returns:
        Path: Caminho do executável do Python do `.venv`.
    """
    py = python_do_venv()

    # Cria o ambiente virtual se ele ainda não existe
    if not py.exists():
        print(f"Criando o ambiente virtual em {VENV} ...")

        try:
            venv.create(VENV, with_pip=True, clear=True)
        except (subprocess.CalledProcessError, ImportError) as erro:
            sys.exit(
                f"Não foi possível criar o ambiente virtual ({erro}).\n"
                "No Debian/Ubuntu instale o pacote 'python3-venv' "
                "(por exemplo: sudo apt install python3-venv) e tente "
                "de novo.")

    # Instala as dependências se for a primeira vez ou se o
    # requirements.txt mudou
    if not MARCA.exists() or MARCA.read_text() != REQUISITOS.read_text():
        print("Instalando as dependências (numpy, scipy, matplotlib) ...")
        subprocess.run(
            [str(py), "-m", "pip", "install", "-r", str(REQUISITOS)],
            check=True)
        MARCA.write_text(REQUISITOS.read_text())

    return py


# -------------------------------------------------------------------- #
# Execução
# -------------------------------------------------------------------- #
def main() -> int:
    """Interpreta o comando recebido, prepara o ambiente e executa o
     script correspondente.

    Returns:
        int: Código de saída (0 em caso de sucesso, 1 para uso
         incorreto, ou o código do script executado).
    """
    # Comando ausente ou desconhecido: mostra o modo de uso
    if len(sys.argv) < 2 or sys.argv[1] not in (*SCRIPTS, "setup"):
        print(__doc__)
        return 1

    py = preparar_ambiente()
    comando = sys.argv[1]

    # "setup" apenas prepara o ambiente
    if comando == "setup":
        print("Ambiente pronto.")
        return 0

    # Executa o script no Python do .venv, repassando os argumentos
    # restantes. O cwd é a raiz do projeto, para que a pasta results/
    # seja criada sempre no mesmo lugar.
    return subprocess.run(
        [str(py), str(RAIZ / SCRIPTS[comando]), *sys.argv[2:]],
        cwd=RAIZ).returncode


if __name__ == "__main__":
    sys.exit(main())
