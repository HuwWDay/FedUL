import torch
from torch import nn, optim
from collections import OrderedDict
import flwr as fl
from flwr.client import ClientApp

from .task import prepare_data, train, test, DigitModel, args

# Load data once for all clients (in a real distributed setup, each client loads its own)
train_loaders, val_loaders, test_loaders, prior_test, client_priors_corr, client_Pi = prepare_data()

class FedULClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device, Pi, priors_corr, prior_test):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.Pi = Pi
        self.priors_corr = priors_corr
        self.prior_test = prior_test
        self.loss_fun = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = optim.Adam(params=self.model.parameters(), lr=args.lr)
        
        for _ in range(args.wk_iters):
            train_loss = train(
                self.model, self.train_loader, optimizer, self.loss_fun, 
                self.device, self.Pi, self.priors_corr, self.prior_test
            )
            
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"loss": train_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        error_rate = test(self.model, self.val_loader, self.device)
        return float(error_rate), len(self.val_loader.dataset), {"error_rate": float(error_rate)}

def client_fn(context: fl.common.Context):
    # Flower assigns a node_id to each client instance
    client_idx = int(context.node_config["partition-id"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DigitModel(class_num=args.classnum).to(device)

    return FedULClient(
        model=model,
        train_loader=train_loaders[client_idx],
        val_loader=val_loaders[client_idx],
        device=device,
        Pi=client_Pi[client_idx],
        priors_corr=client_priors_corr[client_idx],
        prior_test=prior_test
    ).to_client()

# Create the ClientApp
app = ClientApp(client_fn=client_fn)