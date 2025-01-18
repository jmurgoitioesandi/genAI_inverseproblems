import argparse, textwrap

formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=50)


def cla():
    parser = argparse.ArgumentParser(
        description="list of arguments", formatter_class=formatter
    )

    # Data parameters
    parser.add_argument(
        "--dataset_directory",
        type=str,
        default="/scratch1/murgoiti/Datasets/Runze_Phantoms_dataset_for_PyTorch_1em7noise",
        help=textwrap.dedent("""Data file containing training data pairs"""),
    )
    parser.add_argument(
        "--saving_dir",
        type=str,
        default="cWGAN_Runze_Phantoms_110223",
        help=textwrap.dedent("""Directory to save files"""),
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=3000,
        help=textwrap.dedent(
            """Number of training samples to use. Cannot be more than that available."""
        ),
    )
    parser.add_argument(
        "--n_val",
        type=int,
        default=1000,
        help=textwrap.dedent(
            """Number of validation samples to use. Cannot be more than that available."""
        ),
    )
    parser.add_argument(
        "--learn_rate",
        type=float,
        default=1e-4,
        help=textwrap.dedent("""Learning rate."""),
    )
    parser.add_argument(
        "--seed_no",
        type=int,
        default=1008,
        help=textwrap.dedent("""Set the random seed"""),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help=textwrap.dedent("""Batch size"""),
    )
    parser.add_argument(
        "--z_dim",
        type=int,
        default=10,
        help=textwrap.dedent("""Dimension of latent vector"""),
    )
    return parser.parse_args()
