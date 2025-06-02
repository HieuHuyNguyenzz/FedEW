from Baseline import Baseline

class FedProx(Baseline):
    def __init__(
        self,
        algorithm: str = "FedProx",
    ) -> None:
        super().__init__(algorithm)

    