import torch
from collections import OrderedDict
import flwr as fl
from flwr.server import ServerApp, ServerConfig, ServerAppComponents

from .task import prepare_data, test, DigitModel, args

# The server also needs the test set for centralized evaluation
_, _, test_loaders, _, _, _ = prepare_data()

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    server_model = DigitModel(class_num=args.classnum).to(device)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0, 
        fraction_evaluate=1.0, 
        min_fit_clients=args.clientnum,
        min_evaluate_clients=args.clientnum,
        min_available_clients=args.clientnum,
        evaluate_fn=get_evaluate_fn(server_model, test_loaders, device)
    )

    config = ServerConfig(num_rounds=args.iters)
    return ServerAppComponents(strategy=strategy, config=config)

# Create the ServerApp
app = ServerApp(server_fn=server_fn)