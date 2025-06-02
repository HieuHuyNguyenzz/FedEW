from model import get_model
from utils import test, get_parameters, set_parameters, compute_entropy
from DataProcessing import data_preprocessing
from Strategy.Baseline import FedAvg, FedProx, FedImp
from Client import Baseline_Client, Baseline_Client
from ClientManager import ClientManager
import flwr as fl
from flwr.common import ndarrays_to_parameters, Scalar, NDArrays
import torch
from typing import Dict, Optional, Tuple


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_simulation(algorithm: str = "FedAvg", 
                   dataset: str = "MNIST", 
                   num_clients: int = 10,  
                   learning_rate: float = 0.01, 
                   batch_size: int = 32,
                   num_iids: int = 0,
                   num_rounds: int = 10,
                   alpha: float = 0.01,
                   model_name: str = "CNN1",
                   client_resources: Dict[str, float] = None) -> None:

    # Initialize components
    model = get_model(model_name=model_name).to(DEVICE)
    client_manager = ClientManager()

    # Load and partition data
    trainloaders, testloaders, ids, dist = data_preprocessing(dataset, num_clients, num_iids, alpha, batch_size)

    def evaluate_centralized(
        server_round: int,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        net = model
        set_parameters(net, parameters)
        loss, accuracy, precision, recall, f1_score = test(net, testloaders)
        return loss, {"accuracy": accuracy, "precision":precision, "recall":recall, "f1_score":f1_score}
    

    # Define strategy and client initialization
    if algorithm == "FedAvg":
        strategy = FedAvg(num_rounds=num_rounds, 
                        num_clients=num_clients,
                        current_parameters=current_parameters, 
                        evaluate_fn=evaluate_centralized,
                        learning_rate = learning_rate,
                        client_manager=client_manager,
                        ),

        def client_fn(cid: str) -> Baseline_Client:
            net = model
            trainloader = trainloaders[int(cid)]
            valloader = testloaders
            return Baseline_Client(cid, net, trainloader, valloader).to_client()

    elif algorithm == "FedProx":
        strategy = FedProx(num_rounds=num_rounds, 
                        num_clients=num_clients,
                        current_parameters=current_parameters, 
                        evaluate_fn=evaluate_centralized,
                        learning_rate = learning_rate,
                        client_manager=client_manager,
                        mu=0.01,
                        ),

        def client_fn(cid: str) -> Baseline_Client:
            net = model
            trainloader = trainloaders[int(cid)]
            valloader = testloaders
            return Baseline_Client(cid, net, trainloader, valloader).to_client()

    elif algorithm == "FedImp":
        entropies = [compute_entropy(dist[i]) for i in range(num_clients)]

        strategy = FedImp(num_rounds=num_rounds, 
                        num_clients=num_clients,
                        current_parameters=current_parameters, 
                        evaluate_fn=evaluate_centralized,
                        learning_rate = learning_rate,
                        client_manager=client_manager,
                        Entropy=compute_entropy(model),
                        ),

        def client_fn(cid: str) -> Baseline_Client:
            net = model
            trainloader = trainloaders[int(cid)]
            valloader = testloaders
            entropy = entropies[int(cid)]
            return Baseline_Client(cid, net, trainloader, valloader, entropy).to_client()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


    current_parameters = ndarrays_to_parameters(get_parameters(model))

    fl.simulation.start_simulation(
            client_fn = client_fn,
            num_clients = num_clients,
            config = fl.server.ServerConfig(num_rounds=num_rounds),
            strategy = strategy,
            current_parameters = current_parameters,
            client_resources = client_resources
    )
