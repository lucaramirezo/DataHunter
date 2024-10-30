from argparse import Namespace
from queue import Queue
from typing import Set


class Crawler:
    """Clase que representa un Crawler"""

    def __init__(self, args: Namespace):
        self.args = args

    def crawl(self) -> None:
        """Método para crawlear la URL base. `crawl` debe crawlear, desde
        la URL base `args.url`, usando la librería `requests` de Python,
        el número máximo de webs especificado en `args.max_webs`.
        Puedes usar una cola para esto:

        https://docs.python.org/3/library/queue.html#queue.Queue

        Para cada nueva URL que se visite, debe almacenar en el directorio
        `args.output_folder` un fichero .json con, al menos, lo siguiente:

        - "url": URL de la web
        - "text": Contenido completo (en crudo, sin parsear) de la web
        """

        queue = Queue()
        queue.put(self.args.url)
        while ...:
            ...

    def find_urls(self, text: str) -> Set[str]:
        """Método para encontrar URLs de la Universidad Europea en el
        texto de una web. SOLO se deben extraer URLs que aparezcan en
        como valores "href" y que sean de la Universidad, esto es,
        deben empezar por "https://universidadeuropea.com".
        `find_urls` será útil para el proceso de crawling en el método `crawl`

        Args:
            text (str): text de una web
        Returns:
            Set[str]: conjunto de urls (únicas) extraídas de la web
        """
        ...
