import os
# Force PyTorch to use 1 thread to prevent Windows CPU gridlock
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
import copy
from collections import OrderedDict
import flwr as fl

# Import your tasks
from fedULapp.task import prepare_data, train, test, DigitModel

# --- GLOBAL SETTINGS ---
NUM_CLIENTS = 5
NUM_ROUNDS = 3
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.1
BATCH_SIZE = 32

print("🚀 Loading data...")
train_loaders, val_loaders, test_loaders, prior_test, client_priors_corr, client_Pi = prepare_data(
    clientnum=NUM_CLIENTS, 
    batch_size=BATCH_SIZE
)
print("✅ Data loaded successfully.")

# --- CLIENT LOGIC ---
class FedULClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_loader, val_loader, device, Pi, priors_corr, prior_test):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.Pi = Pi
        self.priors_corr = priors_corr
        self.prior_test = prior_test
        self.loss_fun = torch.nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print(f"🟢 Client {self.cid} starting training...")
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=LEARNING_RATE)
        
        train_loss = 0.0
        for _ in range(LOCAL_EPOCHS):
            train_loss = train(
                self.model, self.train_loader, optimizer, self.loss_fun, 
                self.device, self.Pi, self.priors_corr, self.prior_test
            )
        
        print(f"🔴 Client {self.cid} finished. Loss: {train_loss:.4f}")
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"loss": train_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        error_rate = test(self.model, self.val_loader, self.device)
        return float(error_rate), len(self.val_loader.dataset), {"error_rate": float(error_rate)}

def client_fn(cid: str) -> fl.client.Client:
    client_idx = int(cid)
    device = torch.device('cpu') # Forcing CPU to avoid VRAM locks
    model = DigitModel(class_num=10).to(device)
    
    return FedULClient(
        cid=cid,
        model=model,
        train_loader=train_loaders[client_idx],
        val_loader=val_loaders[client_idx],
        device=device,
        Pi=client_Pi[client_idx],
        priors_corr=client_priors_corr[client_idx],
        prior_test=prior_test
    ).to_client()

# --- SERVER LOGIC ---
def get_evaluate_fn(model, test_loaders, device):
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        total_errors = []
        for t_loader in test_loaders:
            err = test(model, t_loader, device)
            total_errors.append(err)
        
        avg_error_rate = sum(total_errors) / len(total_errors)
        print(f"\n🔥 === GLOBAL TEST ERROR (Round {server_round}): {avg_error_rate * 100:.2f} % === 🔥\n")
        return float(avg_error_rate), {"error_rate": float(avg_error_rate)}
    return evaluate


# --- EXECUTION GUARD (The Windows Fix) ---
if __name__ == "__main__":
    print("🤖 Booting up Flower Simulation Engine...")
    
    server_device = torch.device('cpu')
    global_model = DigitModel(class_num=10).to(server_device)
    
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=get_evaluate_fn(global_model, test_loaders, server_device)
    )

    # Start the simulation directly (bypassing flwr run)
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": os.cpu_count(), "num_gpus": 0}
    )