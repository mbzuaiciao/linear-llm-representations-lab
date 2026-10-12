from _common import base_parser, emit, regime_comparison

if __name__ == "__main__":
    args = base_parser("Compare global and local affine regimes").parse_args()
    emit(regime_comparison(args.seed, args.samples), args.output, "results/regime_comparison.json")
