from _common import base_parser, emit, local_linearity

if __name__ == "__main__":
    args = base_parser("Measure first-order approximation error").parse_args()
    emit(local_linearity(args.seed), args.output, "results/local_linearity.json")
