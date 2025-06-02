from Run_simulation import run_simulation
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run_simulation(
    algorithm="FedAvg",
    dataset="MNIST",
    num_clients=10,
    learning_rate=0.01,
    batch_size=32,
    num_iids=0,
    num_rounds=10,
    alpha=0.01,
    model_name="MLP",
    client_resources = {"num_cpus": 1, "num_gpus": 0.1}
)