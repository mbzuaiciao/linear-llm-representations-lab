import { describe, expect, it } from 'vitest';
import {
  calculateLocalVsGlobal,
  evaluateFunction,
  computeGlobalFit,
  generateProbeDataset,
  simulateProbeIntervention,
} from '../src/lib/math-explainer';

describe('Deterministic Mathematical Explainers', () => {
  describe('Explainer 1: Global Affine vs Local Linearization', () => {
    it('evaluates sin and tangent correctly at x0 = 0', () => {
      const { y, dy } = evaluateFunction('sin', 0);
      expect(y).toBe(0);
      expect(dy).toBe(1); // cos(0) = 1
    });

    it('shows relative Taylor error approaching 0 as delta shrinks', () => {
      const dataLarge = calculateLocalVsGlobal('sin', 0.5, 1.0);
      const dataMedium = calculateLocalVsGlobal('sin', 0.5, 0.1);
      const dataTiny = calculateLocalVsGlobal('sin', 0.5, 0.01);

      expect(dataTiny.neighborhood.relativeTaylorError).toBeLessThan(dataMedium.neighborhood.relativeTaylorError);
      expect(dataMedium.neighborhood.relativeTaylorError).toBeLessThan(dataLarge.neighborhood.relativeTaylorError);
      expect(dataTiny.neighborhood.relativeTaylorError).toBeLessThan(0.01);
    });

    it('demonstrates that local tangent outperforms global affine for small delta', () => {
      const result = calculateLocalVsGlobal('sin', 1.0, 0.05);
      expect(result.neighborhood.localTangentMse).toBeLessThan(result.neighborhood.globalAffineMse);
    });

    it('computes global fit over [-3, 3] accurately', () => {
      const { slope, intercept } = computeGlobalFit('sin', -3, 3);
      expect(intercept).toBeCloseTo(0, 3); // odd function
      expect(slope).toBeGreaterThan(0); // positive correlation on [-3, 3]
    });
  });

  describe('Explainer 2: Linear Probe vs Causal Intervention', () => {
    it('generates deterministic toy dataset with correlated features', () => {
      const dataset = generateProbeDataset(42, 100);
      expect(dataset.length).toBe(100);

      // Check that z1 is correlated with label
      const posZ1 = dataset.filter((p) => p.label === 1).map((p) => p.z1);
      const negZ1 = dataset.filter((p) => p.label === -1).map((p) => p.z1);
      const meanPosZ1 = posZ1.reduce((a, b) => a + b, 0) / posZ1.length;
      const meanNegZ1 = negZ1.reduce((a, b) => a + b, 0) / negZ1.length;

      expect(meanPosZ1).toBeGreaterThan(1.0);
      expect(meanNegZ1).toBeLessThan(-1.0);
    });

    it('shows probe achieves high baseline decodability', () => {
      const dataset = generateProbeDataset(42, 100);
      const result = simulateProbeIntervention(dataset, 0, 0);

      expect(result.baselineProbeAccuracy).toBeGreaterThan(0.9);
      expect(result.baselineModelAccuracy).toBeGreaterThan(0.9);
      expect(result.causalModelFlippingRate).toBe(0);
    });

    it('demonstrates intervening on probe direction z1 has ZERO effect on downstream model', () => {
      const dataset = generateProbeDataset(42, 100);
      // Large intervention on z1 (shift by +5.0 or -5.0)
      const resultPos = simulateProbeIntervention(dataset, 5.0, 0);
      expect(resultPos.probePredictionChangeRate).toBeGreaterThan(0.3); // probe flipped
      expect(resultPos.causalModelFlippingRate).toBe(0); // Model did not flip at all!

      const resultNeg = simulateProbeIntervention(dataset, -5.0, 0);
      expect(resultNeg.probePredictionChangeRate).toBeGreaterThan(0.3);
      expect(resultNeg.causalModelFlippingRate).toBe(0); // Model still did not flip!
    });

    it('demonstrates intervening on causal direction z2 flips downstream model', () => {
      const dataset = generateProbeDataset(42, 100);
      // Intervene on causal coordinate z2 by -4.0 (reverses positive points)
      const result = simulateProbeIntervention(dataset, 0, -4.0);
      expect(result.causalModelFlippingRate).toBeGreaterThan(0.4); // Downstream outputs flip!
    });
  });
});
