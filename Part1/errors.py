class ProbsLenError(Exception):
    """Exception raised for errors in the input demand curve.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message='Probs array len not exactly 8'):
        self.message = message
        super().__init__(self.message)
