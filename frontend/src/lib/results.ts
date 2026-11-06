import fs from 'node:fs';
import path from 'node:path';

function findRepoRoot(): string {
  const cwd = process.cwd();
  if (fs.existsSync(path.join(cwd, 'results')) && fs.existsSync(path.join(cwd, 'figures'))) {
    return cwd;
  }
  const parent = path.resolve(cwd, '..');
  if (fs.existsSync(path.join(parent, 'results')) && fs.existsSync(path.join(parent, 'figures'))) {
    return parent;
  }
  return path.resolve(cwd, '..');
}

export interface ModelMetricRow {
  name: string;
  interpMse: number;
  interpR2: number;
  shiftMse: number;
  shiftR2: number;
  top1Agreement?: number;
  parameters: number;
}

export interface CapstoneArtifact {
  seed: number;
  samples: number;
  model: string;
  layerPair: [number, number];
  rows: ModelMetricRow[];
  hasFigure: boolean;
  figurePath: string;
  markdownContent: string;
  provenance: {
    command: string;
    script: string;
    jsonPath: string;
    markdownPath: string;
    modelType: 'pretrained' | 'synthetic';
    notes: string;
  };
}

export interface LocalLinearityArtifact {
  seed: number;
  sizes: number[];
  relativeErrors: number[];
  naturalTransitionScale: number;
  hasFigure: boolean;
  figurePath: string;
  provenance: {
    command: string;
    script: string;
    jsonPath: string;
  };
}

export interface RankSweepArtifact {
  seed: number;
  rows: Array<{
    rank: number;
    parameters: number;
    mse: number;
    cosine: number;
    r2: number;
  }>;
  hasFigure: boolean;
  figurePath: string;
  provenance: {
    command: string;
    script: string;
    jsonPath: string;
  };
}

export interface RegimeComparisonArtifact {
  seed: number;
  models: Record<
    string,
    {
      mse: number;
      cosine: number;
      r2: number;
      parameters: number;
    }
  >;
  hasFigure: boolean;
  figurePath: string;
  provenance: {
    command: string;
    script: string;
    jsonPath: string;
  };
}

export interface TrajectoryAnalysisArtifact {
  seed: number;
  warning?: string;
  layers: Array<{
    layer: number;
    mean_step: number;
    mean_turn_angle: number;
    mean_displacement_cosine: number;
  }>;
  hasFigure: boolean;
  figurePath: string;
  provenance: {
    command: string;
    script: string;
    jsonPath: string;
  };
}

export interface LayerLinearityArtifact {
  seed: number;
  model: string;
  layers: Array<{
    layer_pair: [number, number];
    mse: number;
    cosine: number;
    r2: number;
    parameters: number;
  }>;
  hasFigure: boolean;
  figurePath: string;
  provenance: {
    command: string;
    script: string;
    jsonPath: string;
  };
}

export interface ExperimentSummary {
  id: string;
  title: string;
  subtitle: string;
  script: string;
  command: string;
  modelType: 'pretrained' | 'synthetic';
  evaluationType: 'in-distribution' | 'distribution-shift' | 'comparative' | 'descriptive';
  status: 'committed' | 'missing';
  hasFigure: boolean;
  figureUrl?: string;
  primaryCaveat: string;
}

export function getCapstoneArtifact(): CapstoneArtifact | null {
  const root = findRepoRoot();
  const jsonPath = path.join(root, 'results', 'capstone.json');
  const mdPath = path.join(root, 'results', 'capstone.md');
  const figPath = path.join(root, 'figures', 'capstone.png');

  if (!fs.existsSync(jsonPath)) {
    return null;
  }

  try {
    const raw = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
    const markdownContent = fs.existsSync(mdPath) ? fs.readFileSync(mdPath, 'utf-8') : '';
    const hasFigure = fs.existsSync(figPath);

    // Extract models from interpolation and distribution_shift
    const interp = raw.interpolation || {};
    const shift = raw.distribution_shift || {};
    const modelKeys = ['identity', 'mean', 'affine', 'rank_1', 'local_2', 'mlp'];

    const rows: ModelMetricRow[] = modelKeys.map((key) => {
      const iData = interp[key] || {};
      const sData = shift[key] || {};
      return {
        name: key,
        interpMse: iData.mse ?? 0,
        interpR2: iData.r2 ?? 0,
        shiftMse: sData.mse ?? 0,
        shiftR2: sData.r2 ?? 0,
        top1Agreement: sData.downstream_top1_agreement ?? iData.downstream_top1_agreement,
        parameters: iData.parameters ?? sData.parameters ?? 0,
      };
    });

    return {
      seed: raw.seed ?? 0,
      samples: raw.samples ?? 60,
      model: raw.model ?? 'sshleifer/tiny-gpt2',
      layerPair: raw.layer_pair ?? [0, 2],
      rows,
      hasFigure,
      figurePath: '/figures/capstone.png',
      markdownContent,
      provenance: {
        command: 'uv run python experiments/run_capstone.py --transformer --samples 60',
        script: 'experiments/run_capstone.py',
        jsonPath: 'results/capstone.json',
        markdownPath: 'results/capstone.md',
        modelType: 'pretrained',
        notes: 'Pretrained sshleifer/tiny-gpt2 last-token residual states. Held-out declaratives vs question prompts.',
      },
    };
  } catch {
    return null;
  }
}

export function getLocalLinearityArtifact(): LocalLinearityArtifact | null {
  const root = findRepoRoot();
  const jsonPath = path.join(root, 'results', 'local_linearity.json');
  const figPath = path.join(root, 'figures', 'local_linearity.png');

  if (!fs.existsSync(jsonPath)) {
    return null;
  }

  try {
    const raw = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
    return {
      seed: raw.seed ?? 0,
      sizes: raw.sizes ?? [],
      relativeErrors: raw.relative_errors ?? [],
      naturalTransitionScale: raw.natural_transition_scale ?? 0,
      hasFigure: fs.existsSync(figPath),
      figurePath: '/figures/local_linearity.png',
      provenance: {
        command: 'uv run python experiments/run_local_linearity.py',
        script: 'experiments/run_local_linearity.py',
        jsonPath: 'results/local_linearity.json',
      },
    };
  } catch {
    return null;
  }
}

export function getRankSweepArtifact(): RankSweepArtifact | null {
  const root = findRepoRoot();
  const jsonPath = path.join(root, 'results', 'rank_sweep.json');
  const figPath = path.join(root, 'figures', 'rank_sweep.png');

  if (!fs.existsSync(jsonPath)) {
    return null;
  }

  try {
    const raw = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
    return {
      seed: raw.seed ?? 0,
      rows: raw.rows ?? [],
      hasFigure: fs.existsSync(figPath),
      figurePath: '/figures/rank_sweep.png',
      provenance: {
        command: 'uv run python experiments/run_rank_sweep.py',
        script: 'experiments/run_rank_sweep.py',
        jsonPath: 'results/rank_sweep.json',
      },
    };
  } catch {
    return null;
  }
}

export function getRegimeComparisonArtifact(): RegimeComparisonArtifact | null {
  const root = findRepoRoot();
  const jsonPath = path.join(root, 'results', 'regime_comparison.json');
  const figPath = path.join(root, 'figures', 'regime_comparison.png');

  if (!fs.existsSync(jsonPath)) {
    return null;
  }

  try {
    const raw = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
    return {
      seed: raw.seed ?? 0,
      models: raw.models ?? {},
      hasFigure: fs.existsSync(figPath),
      figurePath: '/figures/regime_comparison.png',
      provenance: {
        command: 'uv run python experiments/run_regime_comparison.py',
        script: 'experiments/run_regime_comparison.py',
        jsonPath: 'results/regime_comparison.json',
      },
    };
  } catch {
    return null;
  }
}

export function getTrajectoryAnalysisArtifact(): TrajectoryAnalysisArtifact | null {
  const root = findRepoRoot();
  const jsonPath = path.join(root, 'results', 'trajectory_analysis.json');
  const figPath = path.join(root, 'figures', 'trajectory_analysis.png');

  if (!fs.existsSync(jsonPath)) {
    return null;
  }

  try {
    const raw = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
    return {
      seed: raw.seed ?? 0,
      warning: raw.warning,
      layers: raw.layers ?? [],
      hasFigure: fs.existsSync(figPath),
      figurePath: '/figures/trajectory_analysis.png',
      provenance: {
        command: 'uv run python experiments/run_trajectory_analysis.py',
        script: 'experiments/run_trajectory_analysis.py',
        jsonPath: 'results/trajectory_analysis.json',
      },
    };
  } catch {
    return null;
  }
}

export function getLayerLinearityArtifact(): LayerLinearityArtifact | null {
  const root = findRepoRoot();
  const jsonPath = path.join(root, 'results', 'layer_linearity.json');
  const figPath = path.join(root, 'figures', 'layer_linearity.png');

  if (!fs.existsSync(jsonPath)) {
    return null;
  }

  try {
    const raw = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
    return {
      seed: raw.seed ?? 0,
      model: raw.model ?? 'synthetic',
      layers: raw.layers ?? [],
      hasFigure: fs.existsSync(figPath),
      figurePath: '/figures/layer_linearity.png',
      provenance: {
        command: 'uv run python experiments/run_layer_linearity.py --synthetic',
        script: 'experiments/run_layer_linearity.py',
        jsonPath: 'results/layer_linearity.json',
      },
    };
  } catch {
    return null;
  }
}

export function getAllExperiments(): ExperimentSummary[] {
  const root = findRepoRoot();

  const experiments: ExperimentSummary[] = [
    {
      id: 'capstone',
      title: 'Capstone Evaluation: Generalization & Distribution Shift',
      subtitle: 'Compares affine, rank-1, local regimes, and MLP models on tiny-gpt2 residual states',
      script: 'experiments/run_capstone.py',
      command: 'uv run python experiments/run_capstone.py --transformer --samples 60',
      modelType: 'pretrained',
      evaluationType: 'distribution-shift',
      status: fs.existsSync(path.join(root, 'results', 'capstone.json')) ? 'committed' : 'missing',
      hasFigure: fs.existsSync(path.join(root, 'figures', 'capstone.png')),
      figureUrl: '/figures/capstone.png',
      primaryCaveat: 'Tested on tiny-gpt2 (hidden dim 2). Rank-1 is almost full capacity. Template leakage can inflate fits.',
    },
    {
      id: 'rank_sweep',
      title: 'Rank Sweep: Parameter Count vs Generalization',
      subtitle: 'Measures low-rank approximation capacity on synthetic high-dimensional dynamics',
      script: 'experiments/run_rank_sweep.py',
      command: 'uv run python experiments/run_rank_sweep.py',
      modelType: 'synthetic',
      evaluationType: 'in-distribution',
      status: fs.existsSync(path.join(root, 'results', 'rank_sweep.json')) ? 'committed' : 'missing',
      hasFigure: fs.existsSync(path.join(root, 'figures', 'rank_sweep.png')),
      figureUrl: '/figures/rank_sweep.png',
      primaryCaveat: 'Synthetic linear dynamics; held-out compressibility is evidence of rank, not causal representation in LLMs.',
    },
    {
      id: 'trajectory_analysis',
      title: 'Token Trajectory Straightening Across Layers',
      subtitle: 'Inspects step lengths, turn angles, and displacement cosines across depth',
      script: 'experiments/run_trajectory_analysis.py',
      command: 'uv run python experiments/run_trajectory_analysis.py',
      modelType: 'synthetic',
      evaluationType: 'descriptive',
      status: fs.existsSync(path.join(root, 'results', 'trajectory_analysis.json')) ? 'committed' : 'missing',
      hasFigure: fs.existsSync(path.join(root, 'figures', 'trajectory_analysis.png')),
      figureUrl: '/figures/trajectory_analysis.png',
      primaryCaveat: 'Summary curvature and PCA projections are descriptive qualitative statistics, not causal proof.',
    },
    {
      id: 'local_linearity',
      title: 'Local Linearization & Relative Taylor Error',
      subtitle: 'Directional Jacobian-vector product error across perturbation scales',
      script: 'experiments/run_local_linearity.py',
      command: 'uv run python experiments/run_local_linearity.py',
      modelType: 'synthetic',
      evaluationType: 'in-distribution',
      status: fs.existsSync(path.join(root, 'results', 'local_linearity.json')) ? 'committed' : 'missing',
      hasFigure: fs.existsSync(path.join(root, 'figures', 'local_linearity.png')),
      figureUrl: '/figures/local_linearity.png',
      primaryCaveat: 'Taylor error must vanish as perturbation goes to zero. The empirical question is whether the useful radius is non-trivial.',
    },
    {
      id: 'regime_comparison',
      title: 'Piecewise Local Regimes vs Global Affine Mapping',
      subtitle: 'Compares k-means assigned local models against global least squares',
      script: 'experiments/run_regime_comparison.py',
      command: 'uv run python experiments/run_regime_comparison.py',
      modelType: 'synthetic',
      evaluationType: 'comparative',
      status: fs.existsSync(path.join(root, 'results', 'regime_comparison.json')) ? 'committed' : 'missing',
      hasFigure: fs.existsSync(path.join(root, 'figures', 'regime_comparison.png')),
      figureUrl: '/figures/regime_comparison.png',
      primaryCaveat: 'More regimes add parameters; local-4 and local-8 overfit with higher test error than global affine.',
    },
    {
      id: 'layer_linearity',
      title: 'Layer-to-Layer Affine Predictability',
      subtitle: 'Affine mapping between consecutive transformer/synthetic hidden states',
      script: 'experiments/run_layer_linearity.py',
      command: 'uv run python experiments/run_layer_linearity.py --synthetic',
      modelType: 'synthetic',
      evaluationType: 'in-distribution',
      status: fs.existsSync(path.join(root, 'results', 'layer_linearity.json')) ? 'committed' : 'missing',
      hasFigure: fs.existsSync(path.join(root, 'figures', 'layer_linearity.png')),
      figureUrl: '/figures/layer_linearity.png',
      primaryCaveat: 'Residual stream identity baseline must be reported alongside any affine fit.',
    },
  ];

  return experiments;
}
