import pickle as pkl
from argparse import Namespace
from dataclasses import dataclass
from time import time
from typing import Dict, List

from ..indexer.indexer import Index


@dataclass
class Result:
    """Clase que contendrá un resultado de búsqueda"""

    url: str
    snippet: str

    def __str__(self) -> str:
        print(f"{self.url} -> {self.snippet}")


class Retriever:
    """Clase que representa un recuperador"""

    def __init__(self, args: Namespace):
        self.args = args
        self.index = self.load_index()

    def search_query(self, query: str) -> List[Result]:
        """Método para resolver una query.
        Este método debe ser capaz, al menos, de resolver consultas como:
        "grado AND NOT master OR docencia", con un procesado de izquierda
        a derecha. Por simplicidad, podéis asumir que los operadores AND,
        NOT y OR siempre estarán en mayúsculas.

        Ejemplo para "grado AND NOT master OR docencia":

        posting["grado"] = [1,2,3] (doc ids que tienen "grado")
        NOT posting["master"] = [3, 4, 5] (doc ids que no tienen "master")
        posting["docencia"] = [6] (doc ids que tienen docencia)

        [1, 2, 3] AND [3, 4, 5] OR [6] = [3] OR [6] = [3, 6]

        Args:
            query (str): consulta a resolver
        Returns:
            List[Result]: lista de resultados que cumplen la consulta
        """
        ...

    def search_from_file(self, fname: str) -> Dict[str, List[Result]]:
        """Método para hacer consultas desde fichero.
        Debe ser un fichero de texto con una consulta por línea.

        Args:
            fname (str): ruta del fichero con consultas
        Return:
            Dict[str, List[Result]]: diccionario con resultados de cada consulta
        """
        with open(fname, "r") as fr:
            ts = time()
            ...
            te = time()
            print(f"Time to solve {n_queries}: {te-ts}")

    def load_index(self) -> Index:
        """Método para cargar un índice invertido desde disco."""
        with open(self.args.index_file, "rb") as fr:
            return pkl.load(fr)

    def _and_(self, posting_a: List[int], posting_b: List[int]) -> List[int]:
        """Método para calcular la intersección de dos posting lists.
        Será necesario para resolver queries que incluyan "A AND B"
        en `search_query`.

        Args:
            posting_a (List[int]): una posting list
            posting_b (List[int]): otra posting list
        Returns:
            List[int]: posting list de la intersección
        """
        ...

    def _or_(self, posting_a: List[int], posting_b: List[int]) -> List[int]:
        """Método para calcular la unión de dos posting lists.
        Será necesario para resolver queries que incluyan "A OR B"
        en `search_query`.

        Args:
            posting_a (List[int]): una posting list
            posting_b (List[int]): otra posting list
        Returns:
            List[int]: posting list de la unión
        """
        ...

    def _not_(self, posting_a: List[int]) -> List[int]:
        """Método para calcular el complementario de una posting list.
        Será necesario para resolver queries que incluyan "NOT A"
        en `search_query`

        Args:
            posting_a (List[int]): una posting list
        Returns:
            List[int]: complementario de la posting list
        """
        ...
