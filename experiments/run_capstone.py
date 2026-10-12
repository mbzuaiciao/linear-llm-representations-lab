from _common import base_parser, capstone, emit, transformer_capstone, write_capstone_report

from linear_llm_lab.models import DEFAULT_MODEL

if __name__ == "__main__":
    parser = base_parser("Run the low-complexity linear-regime capstone")
    parser.add_argument("--transformer", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    result = (
        transformer_capstone(args.seed, args.model, args.samples)
        if args.transformer
        else capstone(args.seed, args.samples)
    )
    emit(result, args.output, "results/capstone.json")
    if args.output is None:
        write_capstone_report(result)
