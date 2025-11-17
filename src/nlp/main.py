import logging

from nlp.env import PACKAGE_DIR

logger = logging.getLogger(__name__)


def main() -> None:
    """Sample entry point."""
    logging.basicConfig(level=logging.INFO)
    logger.info('Hello, World!')
    logger.info('PACKAGE_DIR=%s', PACKAGE_DIR)


if __name__ == '__main__':
    main()
