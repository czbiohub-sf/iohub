import logging

import iohub  # noqa: F401


def test_external_logging_level(caplog):
    with caplog.at_level(logging.DEBUG):
        iohub_logger = logging.getLogger("iohub")
        assert iohub_logger.getEffectiveLevel() == logging.INFO
        other_logger = logging.getLogger(__name__)
        assert other_logger.getEffectiveLevel() == logging.DEBUG
