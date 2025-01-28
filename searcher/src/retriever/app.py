from argparse import ArgumentParser

from .retriever import Retriever


def parse_args():
    parser = ArgumentParser(
        prog="Retriever",
        description="Script para ejecutar el retriever. El retriever recibe"
        " , como mínimo, el fichero donde se guarda el índice invertido.",
    )

    parser.add_argument(
        "-i",
        "--index-file",
        type=str,
        help="Ruta del fichero con el índice invertido",
        required=True,
    )

    parser.add_argument(
        "-q", "--query", type=str, help="Query a resolver", required=False
    )

    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Ruta al fichero de texto con una query por línea",
        required=False,
    )

    # Añade un argumento para elegir el método de ranking
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        choices=["tfidf", "embedding", "combined"],
        default="combined",
        help=(
            "Método de ranking a utilizar. Opciones: 'tfidf', 'embedding', 'combined'."
            " Por defecto: 'combined'."
        ),
    )

    # Añade un argumento para ajustar el peso en el ranking combinado
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.5,
        help=(
            "Peso del método TF-IDF en el ranking combinado. Solo se aplica si el método es 'combined'."
            " Valor por defecto: 0.5."
        ),
    )

    args = parser.parse_args()
    if not args.query and not args.file:
        parser.error(
            "Debes introducir una query (-q) o un fichero (-f) con queries."
        )
    if args.query and args.file:
        parser.error(
            "Introduce solo una query (-q) o un fichero (-f), no ambos."
        )
    return args


if __name__ == "__main__":
    args = parse_args()
    retriever = Retriever(args)
    if args.query:
        # Pasa el método y alpha al realizar la búsqueda
        retriever.search_query(args.query, method=args.method, alpha=args.alpha)
    elif args.file:
        retriever.search_from_file(args.file, method=args.method, alpha=args.alpha)
