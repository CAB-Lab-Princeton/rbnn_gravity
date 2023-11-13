import argparse
from rbnn_gravity.archive.train import run_experiment
from rbnn_gravity.configuration import config


def get_parser(parser):
    parser.add_argument(
        "--gpuid",
        type=int,
        default=0,
        help="The default GPU ID to use. Set -1 to use cpu.",
    )
    parser.add_argument("--ngpu", type=int, default=1, help="Number of gpus to use")

    parser.add_argument(
        "--data_dir",
        type=str,
        default=config.paths.input,
        help="set data directory",
        required=True,
    )
    parser.add_argument(
        "--generated_videos_dir", type=str, default=config.paths.output, required=True
    )
    parser.add_argument(
        "--trained_models_dir", type=str, default=config.paths.output, required=True
    )

    parser.add_argument(
        "--n_examples",
        type=int,
        default=config.dataset.n_examples,
        help="set number of examples, default: 10",
    )
    parser.add_argument(
        "--trajectory_len",
        type=int,
        default=config.dataset.trajectory_len,
        help="set trajectory length, default: 10",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1e-3,
        help="set dt, default: 1e-3",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.experiment.batch_size,
        help="set batch_size, default: 16",
    )
    parser.add_argument(
        "--niter",
        type=int,
        default=config.experiment.n_epoch,
        help="set num of iterations, default: 120000",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="set learning rate, default: 1e-3",
    )

    parser.add_argument(
        "--print",
        type=int,
        default=config.experiment.print_every,
        help="set num of iterations, for print",
    )
    parser.add_argument(
        "--save_output",
        type=int,
        default=config.experiment.save_video_every,
        help="set num of iterations, for save video",
    )
    parser.add_argument(
        "--save_model",
        type=int,
        default=config.experiment.save_model_every,
        help="set num of iterations, for save model",
    )

    parser.add_argument(
        "--seed", type=int, default=config.experiment.seed, help="set random seed"
    )

    return parser


def main(*args):
    parser = argparse.ArgumentParser(description=__doc__)
    args = get_parser(parser).parse_args(args)

    run_experiment(args)
            
            
