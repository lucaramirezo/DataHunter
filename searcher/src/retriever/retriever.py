import pickle as pkl
import unicodedata
from argparse import Namespace
from dataclasses import dataclass
from time import time
from typing import Dict, List
from math import log, sqrt
from numpy import dot
from numpy.linalg import norm
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


    def normalize(self, text: str) -> str:
        """Elimina acentos y convierte el texto a minúsculas."""
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        return text.lower()

    def score(self, query: str, doc_id: int) -> float:
        query_terms = self.normalize(query).split()
        doc_tokens = self.index.documents[doc_id].tokens  # Usar tokens preprocesados

        all_terms = set(query_terms + doc_tokens)
        query_vector = []
        doc_vector = []

        for term in all_terms:
            tf_query = query_terms.count(term)
            tf_query = 1 + log(tf_query) if tf_query > 0 else 0
            idf = log((1 + len(self.index.documents)) / (1 + len(self.index.postings.get(term, [])))) + 1
            query_vector.append(tf_query * idf)

            tf_doc = doc_tokens.count(term)
            tf_doc = 1 + log(tf_doc) if tf_doc > 0 else 0
            doc_vector.append(tf_doc * idf)

        query_norm = norm(query_vector)
        doc_norm = norm(doc_vector)
        if query_norm == 0 or doc_norm == 0:
            return 0.0
        return dot(query_vector, doc_vector) / (query_norm * doc_norm)

    def rank_results(self, query: str, doc_ids: List[int], top_n: int = 10) -> List[Result]:
        """Ordena los resultados por relevancia utilizando similitud del coseno.

        Args:
            query (str): Consulta.
            doc_ids (List[int]): Lista de IDs de documentos relevantes.
            top_n (int): Número máximo de resultados a devolver.

        Returns:
            List[Result]: Resultados ordenados por puntuación.
        """
        print("Iniciando ranking de resultados...")  # Depuración
        scored_results = []
        for doc_id in doc_ids:
            score = self.score(query, doc_id)
            print(f"Doc ID: {doc_id}, Puntaje: {score}")  # Depuración
            if score > 0:  # Solo incluir documentos con puntaje positivo
                snippet = self.index.documents[doc_id].text[:200]  # Seleccionar primer fragmento
                scored_results.append((score, Result(
                    url=self.index.documents[doc_id].url,
                    snippet=snippet
                )))

        # Ordenar por puntuación descendente
        scored_results.sort(key=lambda x: x[0], reverse=True)

        if not scored_results:
            print("No hay documentos relevantes después del ranking.")  # Depuración

        return [result for _, result in scored_results[:top_n]]

    def search_query(self, query: str) -> List[Result]:
        """Resuelve una consulta lógica procesando de izquierda a derecha, respetando la precedencia.

        Args:
            query (str): Consulta a resolver.

        Returns:
            List[Result]: Resultados relevantes con sus URLs y fragmentos ordenados.
        """

        def precedence(op: str) -> int:
            """Devuelve precedencia de operadores."""
            if op == "NOT":
                return 3
            elif op == "AND":
                return 2
            elif op == "OR":
                return 1
            return 0

        def apply_operator(operator: str, posting_a: List[int], posting_b: List[int] = None) -> List[int]:
            """Aplica el operador lógico a las posting lists."""
            if operator == "NOT":
                return self._not_(posting_a)
            elif operator == "AND":
                return self._and_(posting_a, posting_b)
            elif operator == "OR":
                return self._or_(posting_a, posting_b)
            return []

        terms = query.split()
        operands = []
        operators = []

        print(f"Procesando consulta: '{query}'")  # Depuración

        for term in terms:
            if term in ["AND", "OR", "NOT"]:
                while operators and precedence(operators[-1]) >= precedence(term):
                    operator = operators.pop()
                    if operator == "NOT":
                        posting_a = operands.pop()
                        result = apply_operator(operator, posting_a)
                    else:
                        posting_b = operands.pop()
                        posting_a = operands.pop()
                        result = apply_operator(operator, posting_a, posting_b)
                    print(f"Aplicando operador {operator}: {result}")  # Depuración
                    operands.append(result)
                operators.append(term)
            else:
                normalized_term = self.normalize(term)
                posting_list = self.index.postings.get(normalized_term, [])
                print(f"Término: '{term}' (normalizado: '{normalized_term}'), Posting List: {posting_list}")
                operands.append(posting_list)

        # Procesar operadores restantes
        while operators:
            operator = operators.pop()
            if operator == "NOT":
                posting_a = operands.pop()
                result = apply_operator(operator, posting_a)
            else:
                posting_b = operands.pop()
                posting_a = operands.pop()
                result = apply_operator(operator, posting_a, posting_b)
            print(f"Aplicando operador {operator}: {result}")  # Depuración
            operands.append(result)

        final_posting_list = operands.pop() if operands else []
        print(f"Posting List Final: {final_posting_list}")  # Depuración

        if not final_posting_list:
            print("No se encontraron documentos relevantes para la consulta.")
            return []

        # Aplicar ranking por relevancia
        ranked_results = self.rank_results(query, final_posting_list)
        print("\nResultados ordenados por relevancia:")
        for result in ranked_results:
            print(result)

        return ranked_results

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