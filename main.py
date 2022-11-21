import prepare_data
import validate_data
import area
from logger import get_logger


log = get_logger(__name__)


def main() -> None:
    log.info("Getting data")
    prepare_data.run()
    log.info("Finished data acquistion")
    log.info("Validating data")
    validate_data.main()
    log.info("Finished data validation")
    log.info("Starting training")

    pass

if __name__ == "__main__":
    main()