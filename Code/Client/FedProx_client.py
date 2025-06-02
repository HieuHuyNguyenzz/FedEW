import Code.utils as utils
import flwr as fl
from .Baseline import Baseline_Client

class FedProx_Client(Baseline_Client):
    def fit(self, parameters, config):
        utils.set_parameters(self.net, parameters)
        loss, accuracy = utils.train(self.net, self.trainloader, learning_rate=config["learning_rate"], proximal_mu=config["proximal_mu"])
        return utils.get_parameters(self.net), len(self.trainloader.sampler), {"loss": loss, "accuracy": accuracy, "id": self.cid}