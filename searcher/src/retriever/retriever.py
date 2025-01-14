import pickle as pkl
import unicodedata
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
        return f"{self.url} -> {self.snippet}"


class Retriever:
    """Clase que representa un recuperador"""

    def __init__(self, args: Namespace):
        self.args = args
        self.index = self.load_index()

    import unicodedata

    def normalize(self, text: str) -> str:
        """Elimina acentos y convierte el texto a minúsculas."""
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        return text.lower()

    def search_query(self, query: str) -> List[Result]:
        """Resuelve una consulta lógica respetando la precedencia de operadores."""

        def precedence(op: str) -> int:
            """Devuelve la precedencia de los operadores lógicos."""
            if op == "NOT":
                return 3
            elif op == "AND":
                return 2
            elif op == "OR":
                return 1
            return 0

        def apply_operator(operators: List[str], operands: List[List[int]]):
            """Aplica un operador lógico a las listas de posting lists en la pila."""
            operator = operators.pop()
            if operator == "NOT":
                posting_a = operands.pop()
                operands.append(self._not_(posting_a))
            else:
                posting_b = operands.pop()
                posting_a = operands.pop()
                if operator == "AND":
                    operands.append(self._and_(posting_a, posting_b))
                elif operator == "OR":
                    operands.append(self._or_(posting_a, posting_b))

        terms = query.split()
        operators = []  # Pila de operadores lógicos
        operands = []  # Pila de posting lists

        for term in terms:
            if term in ["AND", "OR", "NOT"]:
                while operators and precedence(operators[-1]) >= precedence(term):
                    apply_operator(operators, operands)
                operators.append(term)
            else:
                posting_list = self.index.postings.get(self.normalize(term), [])
                print(f"Posting list para '{term}': {posting_list}")  # Depuración
                operands.append(posting_list)

        while operators:
            apply_operator(operators, operands)

        final_posting_list = operands.pop() if operands else []

        results = []
        for doc_id in final_posting_list:
            document = self.index.documents[doc_id]
            snippet = document.text[:200]
            results.append(Result(url=document.url, snippet=snippet))

        if not results:
            print(f"No se encontraron resultados para la consulta '{query}'.")
        else:
            print(f"\nResultados para la consulta '{query}':")
            for result in results:
                print(result)

        return results

    def search_from_file(self, fname: str) -> Dict[str, List[Result]]:
        """Método para hacer consultas desde fichero.
        Debe ser un fichero de texto con una consulta por línea.

        Args:
            fname (str): ruta del fichero con consultas
        Return:
            Dict[str, List[Result]]: diccionario con resultados de cada consulta
        """
        results = {}
        with open(fname, "r") as fr:
            queries = fr.readlines()

        ts = time()
        for query in queries:
            query = query.strip()
            if query:
                results[query] = self.search_query(query)
        te = time()

        print(f"Time to solve {len(queries)} queries: {te - ts:.2f} seconds")
        return results

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
        i, j = 0, 0
        intersection = []
        while i < len(posting_a) and j < len(posting_b):
            if posting_a[i] == posting_b[j]:
                intersection.append(posting_a[i])
                i += 1
                j += 1
            elif posting_a[i] < posting_b[j]:
                i += 1
            else:
                j += 1
        return intersection

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
        i, j = 0, 0
        union = []
        while i < len(posting_a) and j < len(posting_b):
            if posting_a[i] == posting_b[j]:
                union.append(posting_a[i])
                i += 1
                j += 1
            elif posting_a[i] < posting_b[j]:
                union.append(posting_a[i])
                i += 1
            else:
                union.append(posting_b[j])
                j += 1
        # Agregar elementos restantes de ambas listas
        union.extend(posting_a[i:])
        union.extend(posting_b[j:])
        return union

    def _not_(self, posting_a: List[int]) -> List[int]:
        """Método para calcular el complementario de una posting list.
        Será necesario para resolver queries que incluyan "NOT A"
        en `search_query`

        Args:
            posting_a (List[int]): una posting list
        Returns:
            List[int]: complementario de la posting list
        """
        all_docs = set(range(len(self.index.documents)))
        return sorted(list(all_docs - set(posting_a)))