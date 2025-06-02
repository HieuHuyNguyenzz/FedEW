import Code.utils as utils
import flwr as fl


class Baseline_Client(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return utils.get_parameters(self.net)

    def fit(self, parameters, config):
        utils.set_parameters(self.net, parameters)
        loss, accuracy = utils.train(self.net, self.trainloader, learning_rate=config["learning_rate"])
        return utils.get_parameters(self.net), len(self.trainloader.sampler), {"loss": loss, "accuracy": accuracy, "id": self.cid}

    def evaluate(self, parameters, config):
        utils.set_parameters(self.net, parameters)
        loss, accuracy, precision, recall, f1_score = utils.test(self.net, self.valloader)
        return loss, len(self.valloader.sampler), {"accuracy": accuracy, "precision":precision, "recall":recall, "f1_score":f1_score}
    
class FedImp_Client(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, Entropy):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.Entropy = Entropy

    def get_parameters(self, config):
        return utils.get_parameters(self.net)

    def fit(self, parameters, config):
        utils.set_parameters(self.net, parameters)
        loss, accuracy = utils.train(self.net, self.trainloader, learning_rate=config["learning_rate"])
        return utils.get_parameters(self.net), len(self.trainloader.sampler), {"loss": loss, "accuracy": accuracy, "entropy": self.Entropy}

    def evaluate(self, parameters, config):
        utils.set_parameters(self.net, parameters)
        loss, accuracy, precision, recall, f1_score = utils.test(self.net, self.valloader)
        return loss, len(self.valloader.sampler), {"accuracy": accuracy, "precision":precision, "recall":recall, "f1_score":f1_score}