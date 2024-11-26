import requests
from bs4 import BeautifulSoup
from queue import Queue
from typing import Set
from argparse import Namespace


class Crawler:
    """Clase que representa un Crawler"""

    def __init__(self, args: Namespace):
        self.url = args.url
        self.max_webs = args.max_webs
        self.output_folder = args.output_folder
        self.visited = set()  # URLs visitadas
        self.queue = Queue()  # Cola de URLs por visitar

    def crawl(self) -> None:
        """Método para rastrear páginas web."""
        self.queue.put(self.url)

        while not self.queue.empty() and len(self.visited) < self.max_webs:
            current_url = self.queue.get()

            # Evitar procesar URLs repetidas
            if current_url in self.visited:
                continue

            try:
                response = requests.get(current_url, timeout=10)
                response.raise_for_status()  # Manejar errores HTTP

                # Almacenar contenido en archivo JSON
                self.save_page(current_url, response.text)

                # Extraer y añadir nuevas URLs a la cola
                urls = self.find_urls(response.text)
                for url in urls:
                    if url not in self.visited:
                        self.queue.put(url)

                self.visited.add(current_url)
            except requests.RequestException as e:
                print(f"Error al procesar {current_url}: {e}")

    def find_urls(self, text: str) -> Set[str]:
        """Encuentra URLs de la Universidad Europea en el texto."""
        soup = BeautifulSoup(text, "html.parser")
        urls = set()
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if href.startswith("https://universidadeuropea.com"):
                urls.add(href)
        return urls

    def save_page(self, url: str, text: str) -> None:
        """Guarda el contenido de la página en un archivo JSON."""
        os.makedirs(self.output_folder, exist_ok=True)
        filename = os.path.join(self.output_folder, f"{len(self.visited)}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"url": url, "text": text}, f, ensure_ascii=False, indent=4)