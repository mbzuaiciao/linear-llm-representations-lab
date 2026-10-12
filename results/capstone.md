# Capstone results

## Experimental setup

Seed `0`; `60` prompts/samples. pretrained token residual states; trained on short declarative sentences, tested on held-out declaratives and question-form prompts. Pretrained model `sshleifer/tiny-gpt2`, layer pair `[0, 2]`. The interpolation split holds out declarative prompts; the shift split uses questions.

## Quantitative results

| model | interpolation MSE | interpolation R² | shift MSE | shift R² | top-1 agreement | parameters |
|---|---:|---:|---:|---:|---:|---:|
| identity | 0.667183 | -0.0793 | 0.898425 | 0.0332 | 0.382 | 0 |
| mean | 0.618186 | -0.0001 | 1.046202 | -0.1259 | 0.465 | 2 |
| affine | 0.101278 | 0.8362 | 0.131517 | 0.8585 | 0.993 | 6 |
| rank_1 | 0.101278 | 0.8362 | 0.131517 | 0.8585 | 0.993 | 6 |
| local_2 | 0.073122 | 0.8817 | 0.106214 | 0.8857 | 0.993 | 16 |
| mlp | 0.624841 | -0.0108 | 0.996451 | -0.0723 | 0.215 | 162 |

![Capstone comparison](../figures/capstone.png)

## Interpretation

These are measurements on a tiny pretrained model, not a broad claim about transformer families. Compare compact maps with identity, the mean, the full affine map, and the MLP. Top-1 agreement tests preservation through the frozen unembedding but is coarse when several tokens have close logits.

## Failure modes

The tiny model has hidden width two, so an affine map has only six parameters and rank one is nearly full capacity. Repeated prompt templates and shared tokens can inflate generalization; k-means assignments need not be causal regimes; the question shift is narrow; and top-token agreement hides logit calibration errors.

## What would falsify the linear-dynamics hypothesis?

The useful version would be weakened if compact affine or piecewise-affine maps consistently failed to beat identity/mean baselines on held-out transformer activations, lost all advantage under modest shifts, or required nearly full-rank maps while small nonlinear models generalized reliably.

## Follow-up experiments

Repeat on pretrained residual streams across model families; split by prompt templates; measure downstream logits; compare natural with matched off-manifold perturbations; and intervene along predicted update directions.
