import { describe, expect, it } from 'vitest';
import {
  getCapstoneArtifact,
  getLocalLinearityArtifact,
  getRankSweepArtifact,
  getRegimeComparisonArtifact,
  getTrajectoryAnalysisArtifact,
  getLayerLinearityArtifact,
  getAllExperiments,
} from '../src/lib/results';

describe('Results & Artifacts Loader', () => {
  it('faithfully loads capstone.json and matches expected quantitative structure', () => {
    const capstone = getCapstoneArtifact();
    expect(capstone).not.toBeNull();
    if (!capstone) return;

    expect(capstone.seed).toBe(0);
    expect(capstone.samples).toBe(60);
    expect(capstone.model).toBe('sshleifer/tiny-gpt2');
    expect(capstone.layerPair).toEqual([0, 2]);

    const modelNames = capstone.rows.map((r) => r.name);
    expect(modelNames).toEqual(['identity', 'mean', 'affine', 'rank_1', 'local_2', 'mlp']);

    // Check specific committed numbers from results/capstone.json
    const affine = capstone.rows.find((r) => r.name === 'affine');
    expect(affine).toBeDefined();
    expect(affine?.parameters).toBe(6);
    expect(affine?.interpR2).toBeCloseTo(0.836, 2);
    expect(affine?.shiftR2).toBeCloseTo(0.858, 2);

    const mlp = capstone.rows.find((r) => r.name === 'mlp');
    expect(mlp).toBeDefined();
    expect(mlp?.parameters).toBe(162);
    expect(mlp?.interpR2).toBeLessThan(0); // Overfitting failure on small data

    expect(capstone.hasFigure).toBe(true);
    expect(capstone.markdownContent.length).toBeGreaterThan(0);
  });

  it('loads local_linearity.json and verifies Taylor error structure', () => {
    const local = getLocalLinearityArtifact();
    expect(local).not.toBeNull();
    if (!local) return;

    expect(local.sizes.length).toBeGreaterThan(0);
    expect(local.relativeErrors.length).toBe(local.sizes.length);
    expect(local.naturalTransitionScale).toBeGreaterThan(0);
    // Relative error should strictly decrease for smaller perturbation sizes
    expect(local.relativeErrors[0]).toBeLessThan(local.relativeErrors[local.relativeErrors.length - 1]);
  });

  it('loads rank_sweep.json and trajectory_analysis.json', () => {
    const rankSweep = getRankSweepArtifact();
    expect(rankSweep).not.toBeNull();
    expect(rankSweep?.rows.length).toBe(6);

    const trajectory = getTrajectoryAnalysisArtifact();
    expect(trajectory).not.toBeNull();
    expect(trajectory?.layers.length).toBeGreaterThan(0);
    expect(trajectory?.warning).toContain('PCA/summary geometry is descriptive');
  });

  it('handles missing figure or artifact states gracefully without throwing', () => {
    const experiments = getAllExperiments();
    for (const exp of experiments) {
      expect(typeof exp.hasFigure).toBe('boolean');
      expect(typeof exp.status).toBe('string');
      expect(['committed', 'missing']).toContain(exp.status);
    }
  });

  it('loads layer_linearity and regime_comparison correctly', () => {
    const layerLin = getLayerLinearityArtifact();
    expect(layerLin).not.toBeNull();
    expect(layerLin?.layers[0].r2).toBeGreaterThan(0.99);

    const regimes = getRegimeComparisonArtifact();
    expect(regimes).not.toBeNull();
    expect(regimes?.models.global).toBeDefined();
    expect(regimes?.models.local_2).toBeDefined();
  });

  it('provides all 6 experiments with clear caveats and provenance', () => {
    const experiments = getAllExperiments();
    expect(experiments.length).toBe(6);
    for (const exp of experiments) {
      expect(exp.id).toBeDefined();
      expect(exp.title.length).toBeGreaterThan(0);
      expect(exp.command).toContain('uv run python experiments/');
      expect(exp.primaryCaveat.length).toBeGreaterThan(0);
      expect(exp.status).toBe('committed');
    }
  });
});
