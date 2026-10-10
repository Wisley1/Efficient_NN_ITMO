import sys

from baseline import build_parser, train


def parse_args():
    parser = build_parser()
    parser.set_defaults(
        schedule="triangular",
        max_iter=10_000,
        base_lr=0.1,
        max_lr=3.0,
        stepsize=5000,
        bn_maf=0.95,
        run_name="clr",
        log_interval=10,
        resume_hint="python hw2/clr.py --resume",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        train(parse_args())
    except KeyboardInterrupt:
        sys.exit(130)
