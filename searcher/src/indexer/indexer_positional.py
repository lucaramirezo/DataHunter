import os
import json
import pickle as pkl
import re
from time import time

import unicodedata
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Document:
    """Dataclass para representar un documento.
    Cada documento contendrá:
        - id: identificador único de documento.
        - title: título del documento.
        - url: URL del documento.
        - text: texto del documento, parseado y limpio.
        - tokens
    """

    id: int
    title: str
    url: str
    text: str
    tokens: List[str] = field(default_factory=list)


@dataclass
class Index:
    """Dataclass para representar un índice invertido.

    - "postings": diccionario que mapea palabras a listas de índices. E.g.,
                  si la palabra w1 aparece en los documentos con índices
                  d1, d2 y d3, su posting list será [d1, d2, d3].

    - "documents": lista de `Document`.
    """

    postings: Dict[str, List[int]] = field(default_factory=lambda: {})
    documents: List[Document] = field(default_factory=lambda: [])



    def save(self, output_name: str) -> None:
        """Serializa el índice (`self`) en formato binario usando Pickle"""
        with open(output_name, "wb") as fw:
            pkl.dump(self, fw)


@dataclass
class Stats:
    """Dataclass para representar estadísticas del indexador"""

    n_words: int = field(default_factory=lambda: 0)
    n_docs: int = field(default_factory=lambda: 0)
    building_time: float = field(default_factory=lambda: 0.0)

    def __str__(self) -> str:
        return (
            f"\nEstadísticas del Indexador:\n"
            f"--------------------------\n"
            f"Número de documentos indexados: {self.n_docs}\n"
            f"Número de palabras únicas en el índice: {self.n_words}\n"
            f"Tiempo de construcción: {self.building_time:.2f} segundos\n"
        )


class Indexer:
    """Clase que representa un indexador"""

    def __init__(self, args: Namespace):
        self.args = args
        self.index = Index()
        self.stats = Stats()

        # Descargar stopwords en el constructor
        nltk.download('stopwords', quiet=True)
        self.stopwords = set(stopwords.words('spanish'))

    def build_index(self) -> None:
        ts = time()

        for i, filename in enumerate(os.listdir(self.args.input_folder)):
            if filename.endswith(".json"):
                with open(os.path.join(self.args.input_folder, filename), "r", encoding="utf-8") as f:
                    data = json.load(f)

                    # Procesar contenido del documento
                    text = self.parse(data.get("text", ""))
                    text = self.remove_split_symbols(text)
                    text = self.remove_punctuation(text)
                    text = self.remove_elongated_spaces(text)
                    tokens_with_positions = self.tokenize(text)
                    tokens = [token for token, _ in tokens_with_positions]  # Extraer solo los tokens

                    # Crear el objeto Document
                    document = Document(
                        id=i,
                        title=self.get_title(data.get("text", "")),
                        url=data["url"],
                        text=text,
                        tokens=tokens
                    )
                    self.index.documents.append(document)

                    # Construir el índice invertido posicional
                    for token, position in tokens_with_positions:
                        if token not in self.index.postings:
                            self.index.postings[token] = {}
                        if i not in self.index.postings[token]:
                            self.index.postings[token][i] = []
                        self.index.postings[token][i].append(position)

        te = time()
        self.index.save(self.args.output_name)
        self.show_stats(building_time=te - ts)


    def parse(self, text: str) -> str:
        """Método para extraer el texto de un documento.
        Puedes utilizar la librería 'beautifulsoup' para extraer solo
        el texto del bloque principal de una página web (lee el pdf de la
        actividad para más detalles)

        Args:
            text (str): texto de un documento
        Returns:
            str: texto parseado
        """
        soup = BeautifulSoup(text, "html.parser")
        page_div = soup.find("div", class_="page")
        if not page_div:
            return ""

        # Extraer texto de los tags relevantes dentro del bloque
        raw_texts = []
        for tag in page_div.find_all(["h1", "h2", "h3", "b", "i", "p", "a"]):
            if tag.text:
                raw_texts.append(tag.text.strip())

        return " ".join(raw_texts).lower()

    def normalize(self, text: str) -> str:
        """Elimina acentos y convierte el texto a minúsculas.

        Args:
            text (str): Texto a normalizar.
        Returns:
            str: Texto normalizado.
        """
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        return text.lower()

    def tokenize(self, text: str) -> List[Tuple[str, int]]:
        """Tokeniza el texto y devuelve una lista de (término, posición)."""
        text = self.normalize(text)
        tokens_with_positions = []
        for match in re.finditer(r'\b\w+\b', text):
            token = match.group(0)
            position = match.start()  # Inicio del término en el texto
            tokens_with_positions.append((token, position))
        return tokens_with_positions

    def remove_stopwords(self, words: List[str]) -> List[str]:
        """Método para eliminar stopwords después del tokenizado.
        Puedes usar cualquier lista de stopwords, e.g., de NLTK.

        Args:
            words (List[str]): lista de palabras de un documento
        Returns:
            List[str]: lista de palabras del documento, sin stopwords
        """
        return [word for word in words if word not in self.stopwords]

    def remove_punctuation(self, text: str) -> str:
        """Método para eliminar signos de puntuación de un texto:
         < > ¿ ? , ; : . ( ) [ ] " ' ¡ !

        Args:
            text (str): texto de un documento
        Returns:
            str: texto del documento sin signos de puntuación.
        """
        return re.sub(r'[<>¿?,;:.()\[\]"\'¡!]', "", text)

    def remove_elongated_spaces(self, text: str) -> str:
        """Método para eliminar espacios duplicados.
        E.g., "La     Universidad    Europea" --> "La Universidad Europea"

        Args:
            text (str): texto de un documento
        Returns:
            str: texto sin espacios duplicados
        """
        return " ".join(text.split())

    def remove_split_symbols(self, text: str) -> str:
        """Método para eliminar símbolos separadores como
        saltos de línea, retornos de carro y tabuladores.

        Args:
            text (str): texto de un documento
        Returns:
            str: texto sin símbolos separadores
        """
        return text.replace("\n", " ").replace("\t", " ").replace("\r", " ")

    def get_title(self, html: str) -> str:
        """Extrae el título de la página web."""
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        return title_tag.text.strip() if title_tag else "Sin título"

    def show_stats(self, building_time: float) -> None:
        self.stats.building_time = building_time
        self.stats.n_words = len(self.index.postings)
        self.stats.n_docs = len(self.index.documents)
        print(self.stats)
