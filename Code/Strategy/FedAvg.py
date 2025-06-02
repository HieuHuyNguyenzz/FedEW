from Baseline import Baseline

class FedAvg(Baseline):
    def __init__(
        self,
        algorithm: str = "FedAvg",
    ) -> None:
        super().__init__(algorithm)

    