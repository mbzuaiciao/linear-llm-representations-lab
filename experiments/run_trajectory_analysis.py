from _common import base_parser, emit, trajectory

if __name__ == "__main__":
    args = base_parser("Measure token-trajectory geometry").parse_args()
    emit(trajectory(args.seed, args.samples), args.output, "results/trajectory_analysis.json")
