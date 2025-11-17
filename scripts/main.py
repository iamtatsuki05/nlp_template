import logging


def main() -> None:
    """Sample entry point."""
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info('Hello, World!')


if __name__ == '__main__':
    main()
