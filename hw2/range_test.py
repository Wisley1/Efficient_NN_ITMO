import sys

from baseline import build_parser, train


def parse_args():
    parser = build_parser()
    parser.set_defaults(
        schedule="linear",
        max_iter=5000,
        batch_size=1000,
        base_lr=1e-3,
        max_lr=3.0,
        bn_maf=0.95,
        run_name="range",
        log_interval=10,
        resume_hint="python hw2/range_test.py --resume",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        train(parse_args())
    except KeyboardInterrupt:
        sys.exit(130)
