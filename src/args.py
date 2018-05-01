import argparse
# Common arguments for the various train loops


def get_args(gan_type=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_data", type=int, default=1000)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--random_seed", type=float, default=8509)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--lr", type=int, default=0.0001)

    if gan_type == 'FGAN':
        parser.add_argument("--divergence_type",
                            type=str, default='REVERSE_KL')

    args = parser.parse_args()

    return args
