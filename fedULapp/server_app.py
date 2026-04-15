import torch
from collections import OrderedDict
import flwr as fl
from flwr.server import ServerApp, ServerConfig, ServerAppComponents

from .task import prepare_data, test, DigitModel

# Cache the dataset for centralized evaluation
_server_data_cache = None

def get_server_data(config):
    global _server_data_cache
    if _server_data_cache is None:
        batch_size = config.get("batch-size", 32)
        _server_data_cache = prepare_data(batch_size=batch_size)
    return _server_data_cache

def get_evaluate_fn(model, test_loaders, device):
    """Centralized evaluation function to run after every round."""
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        total_errors = []
        for t_loader in test_loaders:
            err = test(model, t_loader, device)
            total_errors.append(err)
        
        avg_error_rate = sum(total_errors) / len(total_errors)
        print(f"\n=== Global Model Test Error (Round {server_round}): {avg_error_rate * 100:.2f} % ===\n")
        return float(avg_error_rate), {"error_rate": float(avg_error_rate)}
    return evaluate

def server_fn(context: fl.common.Context):
    # Extract config from the pyproject.toml
    run_config = context.run_config
    num_rounds = run_config.get("num-server-rounds", 3)
    fraction_evaluate = run_config.get("fraction-evaluate", 0.5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    server_model = DigitModel(class_num=10).to(device)

    # Grab the test loaders specifically for global evaluation
    _, _, test_loaders, _, _, _ = get_server_data(run_config)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0, 
        fraction_evaluate=fraction_evaluate, 
        min_fit_clients=5, # assuming fixed 5 clients
        min_evaluate_clients=5,
        min_available_clients=5,
        evaluate_fn=get_evaluate_fn(server_model, test_loaders, device)
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

# Create the ServerApp
app = ServerApp(server_fn=server_fn)