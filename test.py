import numpy as np
from Pycrafter6500 import pycrafter6500 as pyc
import usb.core
from unittest.mock import MagicMock
usb.core.find = MagicMock(return_value=MagicMock())

pattern = np.zeros((1080, 1920), dtype=np.uint8)
pattern[:, :960] = 1
images = [pattern]
dlp = pyc.dmd()
dlp.stopsequence = MagicMock()
dlp.changemode = MagicMock()
dlp.defsequence(images, [1000000], [False], [0], [1], 0)
dlp.startsequence = MagicMock()
