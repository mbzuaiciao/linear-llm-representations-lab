from _common import base_parser, emit, rank_sweep

if __name__ == "__main__":
    args = base_parser("Compare low-rank map capacity").parse_args()
    emit(rank_sweep(args.seed, args.samples), args.output, "results/rank_sweep.json")
