from .loader import NTruthMLieLoader


class NTruthMLieProcessor:

    def __init__(self, m, n, path):
        self.loader = NTruthMLieLoader(m, n, path)
