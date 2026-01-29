from collections import deque
import numpy as np


class SequenceBuffer:
    def __init__(self, max_len):
        self.buffer = deque(maxlen=max_len)

    def add(self, frame_features):
        self.buffer.append(frame_features)

    def is_full(self):
        return len(self.buffer) == self.buffer.maxlen

    def to_array(self):
        return np.array(self.buffer, dtype=np.float32)
