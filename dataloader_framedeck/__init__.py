from .evimo import Evimo
from .event_dm import EventDataModule
from .log_logger import LoggingLogger
import os

os.environ['AEGNN_DATA_DIR'] = "/mnt2/shaobo"


def by_name(name: str) -> EventDataModule.__class__:
    if name.lower() == "evimo":
        return Evimo
    else:
        raise NotImplementedError(f"Dataset with name {name} is not known!")
