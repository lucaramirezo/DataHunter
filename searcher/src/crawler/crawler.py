import os
import json
import requests
import re
from queue import Queue
from bs4 import BeautifulSoup
from argparse import Namespace


class Crawler:
    """Clase que representa un Crawler."""

    def __init__(self, args: Namespace):
        self.url = args.url
        self.max_webs = args.max_webs
        self.output_folder = args.output_folder
        self.visited = set()
        self.queue = Queue()

    def crawl(self) -> list:
        """Método para rastrear páginas web y devolver resultados."""
        # Limpia la carpeta de salida antes de comenzar
        self.clean_output_folder()
        os.makedirs(self.output_folder, exist_ok=True)
        self.queue.put(self.url)
        pages = []  # Lista para almacenar los resultados

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
            )
        }

        while not self.queue.empty() and len(self.visited) < self.max_webs:
            current_url = self.queue.get()

            if current_url in self.visited:
                continue

            try:
                # Solicitud con cabecera User-Agent
                response = requests.get(current_url, headers=headers, timeout=10)
                response.raise_for_status()

                # Almacenar el contenido y añadir a la lista
                page_data = {
                    "url": current_url,
                    "text": response.text
                }
                pages.append(page_data)
                self.save_page(current_url, response.text)

                # Extraer nuevas URLs y añadirlas a la cola
                urls = self.find_urls(response.text)
                for url in urls:
                    if url not in self.visited:
                        self.queue.put(url)

                self.visited.add(current_url)
            except requests.RequestException as e:
                print(f"Error al procesar {current_url}: {e}")

        return pages

    def clean_output_folder(self):
        """Limpia la carpeta de salida eliminando todos los archivos."""
        if os.path.exists(self.output_folder):
            for filename in os.listdir(self.output_folder):
                file_path = os.path.join(self.output_folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error al eliminar {file_path}: {e}")


    def find_urls(self, text: str) -> set:
        """Encuentra URLs del dominio 'https://universidadeuropea.com'."""
        urls = set()
        soup = BeautifulSoup(text, "html.parser")
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if href.startswith("https://universidadeuropea.com"):
                urls.add(href)
        return urls

    def save_page(self, url: str, text: str) -> None:
        """Guarda el contenido de la página en un archivo JSON."""
        filename = os.path.join(self.output_folder, f"{len(self.visited)}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"url": url, "text": text}, f, ensure_ascii=False, indent=4)
