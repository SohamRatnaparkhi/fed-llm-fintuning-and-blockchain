"""flowertune-llm: A Flower / FlowerTune app."""

import os
from datetime import datetime

import matplotlib.pyplot as plt
from flwr.common import Context, ndarrays_to_parameters
from flwr.common.config import unflatten_dict
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from omegaconf import DictConfig

from flowertune_llm.dataset import replace_keys
from flowertune_llm.models import get_model, get_parameters, set_parameters

# Get function that will be executed by the strategy's evaluate() method
# Here we use it to save global model checkpoints
all_losses = []


def get_evaluate_fn(model_cfg, save_every_round, total_round, save_path):
    """Return an evaluation function for saving global model."""

    def evaluate(server_round: int, parameters, config):
        # Save model
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
           # Init model
            model = get_model(model_cfg)
            set_parameters(model, parameters)

            model.save_pretrained(f"{save_path}/peft_{server_round}")

        if server_round == total_round:
            # Save losses
            with open(f"{save_path}/all_losses.txt", "w") as f:
                for loss in all_losses:
                    f.write(f"{loss}\n")

            # Save losses graph
            y = all_losses
            x = list(range(1, len(y) + 1))

            plt.plot(x, y)
            plt.xlabel("Round")
            plt.ylabel("Loss")

            name = model_cfg.name
            quantization = model_cfg.quantization
            clients = 3

            save_name = "_".join(
                f"{name}_{quantization}_losses_{clients}_clients".split('/'))
            plt.savefig(f"{save_path}/{save_name}.png")

        return 0.0, {}

    return evaluate


def get_on_fit_config(save_path):
    """Return a function that will be used to construct the config that the client's
    fit() method will receive."""

    def fit_config_fn(server_round: int):
        fit_config = {}
        fit_config["current_round"] = server_round
        fit_config["save_path"] = save_path
        return fit_config

    return fit_config_fn


def fit_weighted_average(metrics):
    """Aggregate (federated) evaluation metrics."""
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    loss = sum(losses) / sum(examples)
    all_losses.append(loss)
    return {"train_loss": sum(losses) / sum(examples)}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(os.getcwd(), f"results/{folder_name}")
    os.makedirs(save_path, exist_ok=True)

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    # Get initial model weights
    init_model = get_model(cfg.model)
    init_model_parameters = get_parameters(init_model)
    init_model_parameters = ndarrays_to_parameters(init_model_parameters)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=cfg.strategy.fraction_fit,
        fraction_evaluate=cfg.strategy.fraction_evaluate,
        on_fit_config_fn=get_on_fit_config(save_path),
        fit_metrics_aggregation_fn=fit_weighted_average,
        initial_parameters=init_model_parameters,
        evaluate_fn=get_evaluate_fn(
            cfg.model, cfg.train.save_every_round, num_rounds, save_path
        ),
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Flower ServerApp
app = ServerApp(server_fn=server_fn)
