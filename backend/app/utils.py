import logging
import json
from datetime import datetime
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
  """
  Formats logs as JSON for structured logging.
  """

  def format(self, record: logging.LogRecord) -> str:
    log_record: Dict[str, Any] = {
      "timestamp": datetime.utcnow().isoformat(),
      "level": record.levelname,
      "message": record.getMessage(),
      "logger": record.name
    }

    if hasattr(record, "extra_data"):
      log_record.update(record.extra_data)

    return json.dumps(log_record)

def get_logger(name: str) -> logging.Logger:
  logger = logging.getLogger(name)
  logger.setLevel(logging.INFO)

  if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

  logger.propagate = False

  return logger

"""
Example Log Output:
{
  "timestamp": "2026-01-05T23:10:12.123Z",
  "level": "INFO",
  "message": "Query classified",
  "logger": "langgraph",
  "intent": "eligibility"
}
"""