from Baseline import Baseline
from typing import List, Tuple, Dict, Union, Optional
import numpy as np
from flwr.common import Parameters, Scalar, FitRes, ClientProxy
from flwr.common.typing import parameters_to_ndarrays, ndarrays_to_parameters
from .Aggregate import aggregate

class FedImp(Baseline):
    def __init__(
        self,
        algorithm: str = "FedImp",
        temperature: float = 0.7,
    ) -> None:
        super().__init__(algorithm)
        self.temperature = temperature
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        weights_results = [(parameters_to_ndarrays(fit_res.parameters),
                            fit_res.num_examples * np.exp(fit_res.metrics["entropy"]/self.temperature))
                            for _, fit_res in results]
        print([fit_res.metrics["id"] for _, fit_res in results])
        self.current_parameters = ndarrays_to_parameters(aggregate(weights_results))
        metrics_aggregated = {}

        losses = [fit_res.num_examples * fit_res.metrics["loss"] for _, fit_res in results]
        corrects = [round(fit_res.num_examples * fit_res.metrics["accuracy"]) for _, fit_res in results]
        examples = [fit_res.num_examples for _, fit_res in results]
        loss = sum(losses) / sum(examples)
        accuracy = sum(corrects) / sum(examples)
        print(f"train_loss: {loss} - train_acc: {accuracy}")

        self.result["round"].append(server_round)
        self.result["train_loss"].append(loss)
        self.result["train_accuracy"].append(accuracy)

        return self.current_parameters, metrics_aggregated

    