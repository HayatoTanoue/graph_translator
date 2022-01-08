import os
import argparse
import json

from train_utils import train
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ("true")


def main(config):
    # For fast training.
    cudnn.benchmark = True
    # Create directories if not exist.
    if not os.path.exists(config.parent_dir + "/model"):
        os.makedirs(config.parent_dir + "/model")
    if not os.path.exists(config.parent_dir + "/sample"):
        os.makedirs(config.parent_dir + "/sample")

    # configをjson形式にして保存
    with open(f"{config.parent_dir}/params.json", mode="w") as f:
        json.dump(config.__dict__, f, indent=4)

    train(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument(
        "--c_dim", type=int, default=5, help="dimension of domain labels (1st dataset)"
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=100,
        help="Resize",
    )
    parser.add_argument(
        "--g_conv_dim",
        type=int,
        default=64,
        help="number of conv filters in the first layer of G",
    )
    parser.add_argument(
        "--d_conv_dim",
        type=int,
        default=64,
        help="number of conv filters in the first layer of D",
    )
    parser.add_argument(
        "--g_repeat_num", type=int, default=6, help="number of residual blocks in G"
    )
    parser.add_argument(
        "--d_repeat_num", type=int, default=6, help="number of strided conv layers in D"
    )
    parser.add_argument(
        "--lambda_cls",
        type=float,
        default=1,
        help="weight for domain classification loss",
    )
    parser.add_argument(
        "--lambda_rec", type=float, default=10, help="weight for reconstruction loss"
    )
    parser.add_argument(
        "--lambda_gp", type=float, default=10, help="weight for gradient penalty"
    )

    parser.add_argument(
        "--lambda_edges", type=float, default=1, help="weight for gradient penalty"
    )

    # Training configuration.
    parser.add_argument(
        "--dataset_name", type=str, default="subset1", choices=["subset1", "poisson", "new_poisson", "new_parete"]
    )
    parser.add_argument("--batch_size", type=int, default=16, help="mini-batch size")
    parser.add_argument(
        "--num_iters",
        type=int,
        default=200000,
        help="number of total iterations for training D",
    )
    parser.add_argument(
        "--num_iters_decay",
        type=int,
        default=100000,
        help="number of iterations for decaying lr",
    )
    parser.add_argument(
        "--g_lr", type=float, default=0.0001, help="learning rate for G"
    )
    parser.add_argument(
        "--d_lr", type=float, default=0.0001, help="learning rate for D"
    )
    parser.add_argument(
        "--n_critic", type=int, default=5, help="number of D updates per each G update"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for Adam optimizer"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="beta2 for Adam optimizer"
    )

    # Test configuration.
    parser.add_argument(
        "--test_iters", type=int, default=200000, help="test model from this step"
    )

    # Miscellaneous.
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])

    # Directories.
    parser.add_argument("--parent_dir", type=str, default="./logs/")

    # Step size.
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--sample_step", type=int, default=1000)
    parser.add_argument("--model_save_step", type=int, default=10000)
    parser.add_argument("--lr_update_step", type=int, default=1000)

    config = parser.parse_args()

    print(config)
    main(config)
