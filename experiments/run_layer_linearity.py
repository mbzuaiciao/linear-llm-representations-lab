from _common import base_parser, emit, layer_linearity

from linear_llm_lab.models import DEFAULT_MODEL

if __name__ == "__main__":
    parser = base_parser("Fit held-out affine maps between adjacent layers")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    emit(layer_linearity(args.seed, args.model, args.synthetic), args.output, "results/layer_linearity.json")
