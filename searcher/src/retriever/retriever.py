import pickle as pkl
import unicodedata
from argparse import Namespace
from dataclasses import dataclass
from time import time
import re
from typing import Dict, List
from math import log, sqrt
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
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
        self.index = None  # Inicializar como None
        self.documents = None  # Para facilitar el acceso a documentos
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Inicializa el modelo de embeddings
        self.load_index()  # Carga el índice y asigna los atributos necesarios


    def normalize(self, text: str) -> str:
        """Elimina acentos y convierte el texto a minúsculas."""
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        return text.lower()

    def calculate_embedding_score(self, query: str, doc_id: int) -> float:
        """Calcula la similitud entre el embedding de la consulta y un documento."""
        query_embedding = self.embedding_model.encode(query)
        doc_embedding = self.documents[doc_id].embedding
        return cosine_similarity([query_embedding], [doc_embedding])[0][0]

    from numpy.linalg import norm

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

    def combined_score(self, query: str, doc_id: int, alpha: float = 0.5) -> float:
        """Combina los puntajes TF-IDF y embeddings en un único valor."""
        tfidf_score = self.score(query, doc_id)
        embedding_score = self.calculate_embedding_score(query, doc_id)
        return alpha * tfidf_score + (1 - alpha) * embedding_score

    def rank_results(self, query: str, doc_ids: List[int], top_n: int = 10, method: str = "combined",
                     alpha: float = 0.5) -> List[Result]:
        """Ordena los resultados por relevancia utilizando el método seleccionado.

        Args:
            query (str): Consulta.
            doc_ids (List[int]): Lista de IDs de documentos relevantes.
            top_n (int): Número máximo de resultados a devolver.
            method (str): Método de ranking ("tfidf", "embedding", "combined").
            alpha (float): Peso para el método combinado (solo aplica para "combined").

        Returns:
            List[Result]: Resultados ordenados por puntuación.
        """
        print("Iniciando ranking de resultados...")  # Depuración
        scored_results = []

        for doc_id in doc_ids:
            if method == "tfidf":
                score = self.score(query, doc_id)
            elif method == "embedding":
                score = self.calculate_embedding_score(query, doc_id)
            elif method == "combined":
                score = self.combined_score(query, doc_id, alpha)
            else:
                raise ValueError(f"Método de ranking desconocido: {method}")

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

    def search_query(self, query: str, method: str = "combined", alpha: float = 0.5) -> List[Result]:
        """Resuelve una consulta lógica procesando operadores y paréntesis.

        Args:
            query (str): Consulta a resolver.
            method (str): Método de ranking ("tfidf", "embedding", "combined").
            alpha (float): Peso del método combinado (solo aplica para "combined").

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

        def apply_operator(operators: List[str], operands: List[List[int]]):
            """Aplica un operador lógico a las listas de posting lists en la pila."""
            operator = operators.pop()
            print(f"Aplicando operador: {operator}")  # Depuración
            if operator == "NOT":
                posting_a = operands.pop()
                print(f"Operando para NOT: {posting_a}")  # Depuración
                result = self._not_(posting_a)
                print(f"Resultado NOT: {result}")  # Depuración
                operands.append(result)
            else:
                posting_b = operands.pop()
                posting_a = operands.pop()
                print(f"Operandos para {operator}: {posting_a}, {posting_b}")  # Depuración
                if operator == "AND":
                    result = self._and_(posting_a, posting_b)
                    print(f"Resultado AND: {result}")  # Depuración
                elif operator == "OR":
                    result = self._or_(posting_a, posting_b)
                    print(f"Resultado OR: {result}")  # Depuración
                operands.append(result)

        # Tokenizar términos y operadores, incluyendo paréntesis para búsquedas parentizadas
        tokens = re.findall(r'\(|\)|AND|OR|NOT|\w+', query)

        operators = []
        operands = []

        print(f"Procesando consulta: '{query}'")  # Depuración
        print(f"Tokens identificados: {tokens}")  # Depuración

        for token in tokens:
            if token == "(":
                print(f"Agregando paréntesis de apertura: {token}")  # Depuración
                operators.append(token)
            elif token == ")":
                print(f"Procesando paréntesis de cierre: {token}")  # Depuración
                while operators and operators[-1] != "(":
                    apply_operator(operators, operands)
                operators.pop()  # Eliminar el paréntesis de apertura
            elif token in ["AND", "OR", "NOT"]:
                print(f"Agregando operador lógico: {token}")  # Depuración
                while operators and operators[-1] != "(" and precedence(operators[-1]) >= precedence(token):
                    apply_operator(operators, operands)
                operators.append(token)
            else:
                # Término de consulta: buscar su posting list
                normalized_term = self.normalize(token)
                posting_list = self.index.postings.get(normalized_term, [])
                print(f"Término: '{token}' (normalizado: '{normalized_term}'), Posting List: {posting_list}")
                operands.append(posting_list)

        # Resolver operadores restantes
        print("Resolviendo operadores restantes...")  # Depuración
        while operators:
            apply_operator(operators, operands)

        # La última posting list en operandos es el resultado final
        final_posting_list = operands.pop() if operands else []
        print(f"Posting List Final: {final_posting_list}")  # Depuración

        if not final_posting_list:
            print("No se encontraron documentos relevantes para la consulta.")
            return []

        # Aplicar ranking por relevancia
        print("Iniciando proceso de ranking...")  # Depuración
        ranked_results = self.rank_results(query, final_posting_list, method=method, alpha=alpha)

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

    def load_index(self):
        """Carga el índice invertido y los embeddings."""
        print(f"Cargando índice desde: {self.args.index_file}")  # Depuración
        try:
            with open(self.args.index_file, 'rb') as f:
                self.index = pkl.load(f)
        except FileNotFoundError:
            raise ValueError(f"El archivo de índice {self.args.index_file} no existe.")
        except Exception as e:
            raise ValueError(f"Error al cargar el índice: {e}")

        # Validar que el índice tiene los atributos esperados
        if not hasattr(self.index, 'postings') or not hasattr(self.index, 'documents'):
            raise ValueError("El índice cargado no tiene el formato esperado.")

        self.documents = self.index.documents  # Asignar los documentos a un atributo separado
        print(
            f"Índice cargado con {len(self.documents)} documentos y {len(self.index.postings)} términos.")

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