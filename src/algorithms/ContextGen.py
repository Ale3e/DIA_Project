import numpy as np

class ContextGen:
    def __init__(self, counts, values, price):
        super().__init__(len(counts))
        self.counts = counts
        self.values = values
        self.price = price