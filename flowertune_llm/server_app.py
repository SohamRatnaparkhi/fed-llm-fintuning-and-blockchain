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
LOSS_FILE_PATH = ""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def get_evaluate_fn(model_cfg, save_every_round, total_round, save_path):
    """Return an evaluation function for saving global model."""

    def evaluate(server_round: int, parameters, config):
        # Save model
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
           # Init model
            model, _ = get_model(model_cfg)
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
            print(x)
            print(y)
            losses = [float(loss) for loss in y]
            plt.figure(figsize=(12, 6))
            plt.plot(x, losses)
            plt.xlabel("Round")
            plt.ylabel("Loss")
            plt.title("Losses over Rounds")
            plt.ylim(min(losses) - 0.05, max(losses) + 0.05)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
            # plt.plot(x, y)
            # plt.xlabel("Round")
            # plt.ylabel("Loss")

            # name = model_cfg.name
            # quantization = model_cfg.quantization
            clients = 3
            name = model_cfg.name
            quant = model_cfg.quantization
            if name != "":
                name = name.split('/')[-1]
            save_name = "_".join(
                f"{name}_{quant}_losses_{clients}_clients".split('/'))
            plt.savefig(f"{save_path}/{save_name}.png")

        return 0.001, {}

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
    print("All losses till now")
    print(all_losses)
    print(LOSS_FILE_PATH)
    with open(LOSS_FILE_PATH, "w") as f:
                for loss in all_losses:
                    f.write(f"{loss}\n")
    return {"train_loss": sum(losses) / sum(examples)}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
    dataset = cfg.dataset.name
    if dataset != "":
        dataset = dataset.split("/")[-1]
    model_name = cfg.model.name or ""
    quant = cfg.model.quantization
    if model_name != "":
        model_name = model_name.split("/")[-1]
    folder_name = f"/{model_name}_{quant}" + current_time.strftime("%Y-%m-%d_%H-%M-%S")
    folder_starter = "/home/sohamr/projects/def-ssamet-ab/sohamr/results/" + dataset + "/"
    save_path = folder_starter + folder_name
    os.makedirs(save_path, exist_ok=True)
    global LOSS_FILE_PATH
    LOSS_FILE_PATH = save_path + '/all_losses.txt'
    print(LOSS_FILE_PATH)
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Get initial model weights
    init_model, tokenizer = get_model(cfg.model)
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
