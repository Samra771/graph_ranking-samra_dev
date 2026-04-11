import random
import os

def shuffle_graphs(graphs_number):
    """
    Genera una lista di interi da 0 a graphs_number-1 e la mescola casualmente.

    Args:
        graphs_number (int): Il numero massimo di grafi da mescolare.

    Returns:
        list: Lista di numeri mescolati.
    """
    indices = list(range(graphs_number))  # Crea una lista [0, 1, 2, ..., graphs_number-1]
    random.shuffle(indices)  # Mescola la lista
    return indices

def get_random_sample(lst, percentage):
    """
    Restituisce una percentuale dei valori di una lista.

    Args:
        lst (list): La lista originale.
        percentage (float): La percentuale di elementi da restituire (0-1).

    Returns:
        list: Una sottolista contenente il percentage% degli elementi originali.
    """
    if not (0 <= percentage <= 100):
        raise ValueError("La percentuale deve essere tra 0 e 100.")

    num_elements = int(len(lst) * percentage)  # Calcola quanti elementi prendere
    return random.sample(lst, num_elements)  # Prende num_elements elementi casuali

def split_train_test_matrices(matrices, train_indices):
    """
    Divide una lista di matrici in due gruppi:
    - Matrici con indice appartenente a train_indices
    - Matrici con indice NON appartenente a train_indices

    Args:
        matrices (list): Lista di matrici (es. liste nidificate o tensori).
        train_indices (list or set): Indici delle matrici da includere nel primo gruppo.

    Returns:
        tuple: (matrici_train, matrici_test)
            - matrici_train: Lista di matrici con indice in train_indices.
            - matrici_test: Lista di matrici con indice non in train_indices.
    """
    train_indices = set(train_indices)  # Convertiamo in set per velocità
    matrici_train = [mat for i, mat in enumerate(matrices) if i in train_indices]
    matrici_test = [mat for i, mat in enumerate(matrices) if i not in train_indices]

    return matrici_train, matrici_test

def split_train_test_list(list_in, train_indices):

    """
    Divide una lista di dizionari in due gruppi:
    - Dizionari con indice appartenente a train_indices
    - Dizionari con indice NON appartenente a train_indices

    Args:
        list_in (list): Lista di dizionari.
        train_indices (list or set): Indici dei dizionari da includere nel primo gruppo.

    Returns:
        tuple: (dicts_train, dicts_test)
            - dicts_train: Lista di dizionari con indice in train_indices.
            - dicts_test: Lista di dizionari con indice non in train_indices.
    """
    train_indices = set(train_indices)  # Convertiamo in set per velocità
    dicts_train = [d for i, d in enumerate(list_in) if i in train_indices]
    dicts_test = [d for i, d in enumerate(list_in) if i not in train_indices]

    return dicts_train, dicts_test

def get_files_with_prefix(prefix: str) -> dict[int, str]:
    """
    Legge i file nella cartella corrente e restituisce un dizionario numerato
    con i file che iniziano con il prefisso specificato e con estensione .pth.

    :param prefix: Il prefisso da cercare nei nomi dei file.
    :return: Dizionario numerato {indice: nome_file}.
    """
    files = [f for f in os.listdir('.') if os.path.isfile(f) and f.startswith(prefix) and f.endswith('.pth')]
    return {i + 1: file for i, file in enumerate(files)}


def get_random_graph(n_min, n_max):
    """Restituisce un numero intero casuale compreso tra n_min e n_max."""
    return random.randint(n_min, n_max)