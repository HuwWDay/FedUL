import logging
from flwr.common.logger import log
import torch
from torch import nn, optim
from collections import OrderedDict
import flwr as fl
from flwr.client import ClientApp

from .task import prepare_data, train, test, DigitModel

_data_cache = None

def get_data(config):
    global _data_cache
    if _data_cache is None:
        batch_size = config.get("batch-size", 32)
        _data_cache = prepare_data(batch_size=batch_size)
    return _data_cache

class FedULClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_loader, val_loader, device, Pi, priors_corr, prior_test, lr, local_epochs):
        self.cid = cid # Store the client ID so we know who is talking
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.Pi = Pi
        self.priors_corr = priors_corr
        self.prior_test = prior_test
        self.loss_fun = nn.CrossEntropyLoss()
        self.lr = lr
        self.local_epochs = local_epochs

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)
        
        train_loss = 0.0
        for _ in range(self.local_epochs):
            train_loss = train(
                self.model, self.train_loader, optimizer, self.loss_fun, 
                self.device, self.Pi, self.priors_corr, self.prior_test
            )
        
        # LOG THE TRAINING PROGRESS
        log(logging.INFO, f"Client {self.cid} finished training. Loss: {train_loss:.4f}")
            
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"loss": train_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        error_rate = test(self.model, self.val_loader, self.device)
        return float(error_rate), len(self.val_loader.dataset), {"error_rate": float(error_rate)}


def client_fn(context: fl.common.Context):
    run_config = context.run_config
    lr = run_config.get("learning-rate", 0.1)
    local_epochs = run_config.get("local-epochs", 1)
    
    client_idx = int(context.node_config["partition-id"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DigitModel(class_num=10).to(device)

    train_loaders, val_loaders, _, prior_test, client_priors_corr, client_Pi = get_data(run_config)

    return FedULClient(
        cid=client_idx, # Pass it here
        model=model,
        train_loader=train_loaders[client_idx],
        val_loader=val_loaders[client_idx],
        device=device,
        Pi=client_Pi[client_idx],
        priors_corr=client_priors_corr[client_idx],
        prior_test=prior_test,
        lr=lr,
        local_epochs=local_epochs
    ).to_client()

app = ClientApp(client_fn=client_fn)