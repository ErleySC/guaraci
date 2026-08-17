"""FASE A / A2. O `mae_id` agrupa exatamente as replicas fisicas da mesma
amostra?

O mae_id e' derivado em `dados_io.parse_title` como:
    puro:        f"{cod}-{data}"
    adulterado:  f"{cod}-{data}-{adulterante}{teor:.2f}"

Isso significa que a IDENTIDADE do grupo depende da DATA e do TEOR
formatado com 2 casas. Dois modos de falha decorrem disso:

  GRAVE  — replicas da mesma amostra fisica recebendo mae_id DIFERENTE.
           Acontece se T1/T2/T3 foram medidas em datas distintas, ou se o
           teor gravado difere na 3a casa decimal entre replicas. O
           GroupKFold entao separa "grupos" que na verdade sao a mesma
           amostra -> o vazamento que o projeto existe para impedir
           sobrevive.
           ASSINATURA DETECTAVEL: grupo com n<3 cujas triplicatas sao um
           SUBCONJUNTO de {1,2,3}, e existe outro grupo do mesmo
           cod+adulterante+teor com as triplicatas COMPLEMENTARES.

  BENIGNO — amostras fisicas distintas colapsadas no mesmo mae_id
           (mesma especie, mesma data, mesmo teor). Reduz o n efetivo.
           Conservador, nao infla acuracia.
           ASSINATURA: grupo com n>3, triplicatas repetidas.

Uso:
    python medir_mae_id_agrupamento.py "<pasta com .dx>"

SAIDA AGREGADA de proposito: nunca imprime nome de arquivo nem TITLE
completo. O repositorio e' publico e o dataset e' de terceiro.
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
from guaraci.dados_io import extrair_title_do_dx, parse_title  # noqa: E402


def varrer(raiz: Path):
    grupos = defaultdict(list)   # mae_id -> [triplicata, ...]
    chave_fisica = defaultdict(set)  # (cod, adult, teor2dec) -> {mae_id, ...}
    n_arq = n_sem_title = n_sem_parse = 0

    for caminho in raiz.rglob("*"):
        if caminho.suffix.lower() not in (".dx", ".jdx"):
            continue
        n_arq += 1
        titulo = extrair_title_do_dx(str(caminho))
        if titulo is None:
            n_sem_title += 1
            continue
        info = parse_title(titulo)
        if info is None:
            n_sem_parse += 1
            continue
        mid = info["mae_id"]
        grupos[mid].append(int(info["triplicata"]))
        teor = info["teor"]
        chave_fisica[(info["cod"], info["adulterante"],
                      None if teor is None else round(teor, 2))].add(mid)

    return grupos, chave_fisica, n_arq, n_sem_title, n_sem_parse


def main(raiz: Path):
    grupos, chave_fisica, n_arq, n_sem_title, n_sem_parse = varrer(raiz)

    print(f"Arquivos espectrais varridos : {n_arq}")
    print(f"  sem linha ##TITLE=         : {n_sem_title}")
    print(f"  TITLE nao casou com o regex: {n_sem_parse}")
    print(f"  parseados                  : {n_arq - n_sem_title - n_sem_parse}")
    print(f"Grupos mae_id distintos      : {len(grupos)}\n")

    dist = Counter(len(v) for v in grupos.values())
    print("Distribuicao de replicas por mae_id")
    print(f"{'n_replicas':>11} {'n_grupos':>9}")
    for n in sorted(dist):
        marca = "  <-- esperado" if n == 3 else ""
        print(f"{n:>11} {dist[n]:>9}{marca}")
    print()

    # --- caso GRAVE: replicas da mesma amostra fisica em mae_id distintos ---
    suspeitos = []
    for chave, mids in chave_fisica.items():
        if len(mids) < 2:
            continue
        # grupos incompletos (n<3) sob a MESMA chave fisica cujas
        # triplicatas nao se sobrepoem -> compativel com replicas partidas
        incompletos = [m for m in mids if len(grupos[m]) < 3]
        if len(incompletos) < 2:
            continue
        trips = [set(grupos[m]) for m in incompletos]
        for i in range(len(trips)):
            for j in range(i + 1, len(trips)):
                if not (trips[i] & trips[j]) and len(trips[i] | trips[j]) <= 3:
                    suspeitos.append((chave, len(incompletos)))
                    break
            else:
                continue
            break

    print(f"CASO GRAVE (replicas partidas entre mae_id): "
          f"{len(suspeitos)} chave(s) fisica(s) suspeita(s)")
    if suspeitos:
        print("  amostras fisicas (cod, adulterante, teor) afetadas — "
              "sem nome de arquivo, repo publico:")
        for chave, n in suspeitos[:40]:
            print(f"    cod={chave[0]:>4}  adult={str(chave[1]):>4}  "
                  f"teor={chave[2]}  -> {n} mae_id incompletos")
        if len(suspeitos) > 40:
            print(f"    ... e mais {len(suspeitos)-40}")
    print()

    # --- caso BENIGNO: grupos maiores que 3 ---
    grandes = {m: v for m, v in grupos.items() if len(v) > 3}
    print(f"CASO BENIGNO (mae_id com >3 espectros): {len(grandes)} grupo(s)")
    for m, v in list(grandes.items())[:20]:
        print(f"    n={len(v):>3}  triplicatas={sorted(Counter(v).items())}")
    print()

    # --- puros por especie: alimenta o veredito do A1 (nc>=6?) ---
    puros = defaultdict(set)
    for (cod, adult, _teor), mids in chave_fisica.items():
        if adult is None:
            puros[cod] |= mids
    print("Grupos mae_id PUROS por codigo de especie "
          "(entra no veredito do A1: nc>=6 ativa a virada de fronteira)")
    print(f"{'cod':>5} {'n_mae_id_puros':>15}")
    for cod in sorted(puros, key=lambda c: -len(puros[c])):
        alerta = "  <-- nc>=6" if len(puros[cod]) >= 6 else ""
        print(f"{cod:>5} {len(puros[cod]):>15}{alerta}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    main(Path(sys.argv[1]))
