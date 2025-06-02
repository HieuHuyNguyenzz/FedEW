import Code.utils as utils
import flwr as fl
from .Baseline import Baseline_Client

class FedImp_Client(Baseline_Client):
    def __init__(self, cid, net, trainloader, valloader, entropy):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.entropy = entropy

    def fit(self, parameters, config):
        utils.set_parameters(self.net, parameters)
        loss, accuracy = utils.train(self.net, self.trainloader, learning_rate=config["learning_rate"])
        return utils.get_parameters(self.net), len(self.trainloader.sampler), {"loss": loss, "accuracy": accuracy, "id": self.cid, "entropy": self.entropy}