import argparse, textwrap

formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=50)


def cla():
    parser = argparse.ArgumentParser(
        description="list of arguments", formatter_class=formatter
    )

    # Data parameters
    parser.add_argument(
        "--saving_dir",
        type=str,
        default="110223",
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
    parser.add_argument(
        "--gp_coef",
        type=int,
        default=10,
        help=textwrap.dedent("""Gradient penalty coefficient"""),
    )
    parser.add_argument(
        "--n_critic",
        type=int,
        default=10,
        help=textwrap.dedent("""Number of critic iterations per generator iteration"""),
    )
    parser.add_argument(
        "--problem",
        type=str,
        required=True,
        help="Inverse problem being solved",
    )
    parser.add_argument(
        "--xtype",
        type=str,
        required=True,
        help="Inferred vector or image",
    )
    parser.add_argument(
        "--noiselvl",
        type=float,
        required=True,
        help="Standard deviation of noise",
    )
    parser.add_argument(
        "--im_size",
        type=int,
        default=64,
        help="Size of the image",
    )
    parser.add_argument(
        "--meas_channels",
        type=int,
        default=1,
        help="Number of measurement channels",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="UNET_CIN",
        help="Type of model to use",
    )
    return parser.parse_args()
