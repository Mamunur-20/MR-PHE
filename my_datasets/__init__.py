from enum import Enum
from .wsss_loader import WSSSDataset
#from .breakhis_loader import BREAKHISDataset

from .bracs7class_loader import BRACSDataset
from .crc_loader import CRCDataset
class MyDataset(str, Enum):
    wsss = "wsss"
#    ebhi = "ebhi"  # Use 'breakhis' instead of 'breakhis_loader'
    crc = "crc"

    def __str__(self) -> str:
        return self.value
