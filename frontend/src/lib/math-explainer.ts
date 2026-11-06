/**
 * Deterministic mathematical routines for interactive explainers.
 * Explicitly pedagogical illustrations; not empirical LLM findings.
 */

export interface LocalVsGlobalData {
  fnName: 'sin' | 'tanh';
  x0: number;
  delta: number;
  // Curve points for graphing
  curvePoints: Array<{ x: number; yTrue: number; yTangent: number; yGlobal: number }>;
  // Metrics in neighborhood [x0 - delta, x0 + delta]
  neighborhood: {
    xMin: number;
    xMax: number;
    relativeTaylorError: number;
    localTangentMse: number;
    globalAffineMse: number;
    tangentSlope: number;
    globalSlope: number;
  };
}

export function evaluateFunction(fnName: 'sin' | 'tanh', x: number): { y: number; dy: number } {
  if (fnName === 'sin') {
    return { y: Math.sin(x), dy: Math.cos(x) };
  }
  // tanh(1.5 * x)
  const w = 1.5;
  const th = Math.tanh(w * x);
  return { y: th, dy: w * (1 - th * th) };
}

export function computeGlobalFit(
  fnName: 'sin' | 'tanh',
  xDomainMin = -3,
  xDomainMax = 3,
  steps = 200
): { slope: number; intercept: number } {
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumX2 = 0;
  const stepSize = (xDomainMax - xDomainMin) / (steps - 1);

  for (let i = 0; i < steps; i++) {
    const x = xDomainMin + i * stepSize;
    const { y } = evaluateFunction(fnName, x);
    sumX += x;
    sumY += y;
    sumXY += x * y;
    sumX2 += x * x;
  }

  const slope = (steps * sumXY - sumX * sumY) / (steps * sumX2 - sumX * sumX);
  const intercept = (sumY - slope * sumX) / steps;
  return { slope, intercept };
}

export function calculateLocalVsGlobal(
  fnName: 'sin' | 'tanh',
  x0: number,
  delta: number,
  xDomainMin = -3,
  xDomainMax = 3,
  numGraphPoints = 120
): LocalVsGlobalData {
  const { slope: globalSlope, intercept: globalIntercept } = computeGlobalFit(fnName, xDomainMin, xDomainMax);
  const { y: y0, dy: tangentSlope } = evaluateFunction(fnName, x0);

  // Tangent line: T(x) = y0 + dy * (x - x0)
  const tangentAt = (x: number) => y0 + tangentSlope * (x - x0);
  const globalAt = (x: number) => globalSlope * x + globalIntercept;

  // Graph curve points
  const curvePoints: LocalVsGlobalData['curvePoints'] = [];
  const step = (xDomainMax - xDomainMin) / (numGraphPoints - 1);
  for (let i = 0; i < numGraphPoints; i++) {
    const x = xDomainMin + i * step;
    curvePoints.push({
      x,
      yTrue: evaluateFunction(fnName, x).y,
      yTangent: tangentAt(x),
      yGlobal: globalAt(x),
    });
  }

  // Relative Taylor Error at x0 + delta: |f(x0 + delta) - (y0 + dy * delta)| / delta
  const xPerturbed = x0 + delta;
  const yPerturbed = evaluateFunction(fnName, xPerturbed).y;
  const yPerturbedLinear = y0 + tangentSlope * delta;
  const relativeTaylorError = Math.abs(yPerturbed - yPerturbedLinear) / Math.max(1e-9, Math.abs(delta));

  // Local MSE over sample points in [x0 - delta, x0 + delta]
  const sampleCount = 25;
  const xMin = x0 - delta;
  const xMax = x0 + delta;
  let sumErrTangent = 0;
  let sumErrGlobal = 0;

  for (let i = 0; i < sampleCount; i++) {
    const xs = xMin + (i / (sampleCount - 1)) * (xMax - xMin);
    const yt = evaluateFunction(fnName, xs).y;
    sumErrTangent += Math.pow(yt - tangentAt(xs), 2);
    sumErrGlobal += Math.pow(yt - globalAt(xs), 2);
  }

  return {
    fnName,
    x0,
    delta,
    curvePoints,
    neighborhood: {
      xMin,
      xMax,
      relativeTaylorError,
      localTangentMse: sumErrTangent / sampleCount,
      globalAffineMse: sumErrGlobal / sampleCount,
      tangentSlope,
      globalSlope,
    },
  };
}

export interface ProbePoint {
  id: number;
  label: -1 | 1;
  z1: number; // probe direction / non-causal
  z2: number; // causal direction
}

export function generateProbeDataset(seed = 42, count = 100): ProbePoint[] {
  // Deterministic pseudo-random PRNG (linear congruential generator)
  let state = seed;
  const random = () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296;
  };
  const normal = () => {
    const u1 = Math.max(1e-9, random());
    const u2 = random();
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  };

  const points: ProbePoint[] = [];
  for (let i = 0; i < count; i++) {
    const label: -1 | 1 = i % 2 === 0 ? 1 : -1;
    // z1 strongly correlated with label (mean +/- 2.0)
    const z1 = label * 2.0 + 0.5 * normal();
    // z2 also correlated with label (mean +/- 1.4), which downstream model actually uses
    const z2 = label * 1.4 + 0.6 * normal();
    points.push({ id: i, label, z1, z2 });
  }
  return points;
}

export interface ProbeInterventionResult {
  deltaZ1: number;
  deltaZ2: number;
  baselineProbeAccuracy: number;
  postInterventionProbeAccuracy: number;
  baselineModelAccuracy: number;
  postInterventionModelAccuracy: number;
  causalModelFlippingRate: number; // fraction of model outputs that changed
  probePredictionChangeRate: number; // fraction of probe outputs that changed
}

export function simulateProbeIntervention(
  points: ProbePoint[],
  deltaZ1: number,
  deltaZ2: number
): ProbeInterventionResult {
  // Probe decision boundary: predict sign(z1)
  // Downstream model computation: predict sign(z2) (ignores z1 completely!)

  let baselineProbeCorrect = 0;
  let postProbeCorrect = 0;
  let baselineModelCorrect = 0;
  let postModelCorrect = 0;
  let modelFlipped = 0;
  let probeFlipped = 0;

  for (const pt of points) {
    const origProbe = pt.z1 >= 0 ? 1 : -1;
    const origModel = pt.z2 >= 0 ? 1 : -1;

    if (origProbe === pt.label) baselineProbeCorrect++;
    if (origModel === pt.label) baselineModelCorrect++;

    const newZ1 = pt.z1 + deltaZ1;
    const newZ2 = pt.z2 + deltaZ2;

    const newProbe = newZ1 >= 0 ? 1 : -1;
    const newModel = newZ2 >= 0 ? 1 : -1;

    if (newProbe === pt.label) postProbeCorrect++;
    if (newModel === pt.label) postModelCorrect++;

    if (newModel !== origModel) modelFlipped++;
    if (newProbe !== origProbe) probeFlipped++;
  }

  const n = points.length;
  return {
    deltaZ1,
    deltaZ2,
    baselineProbeAccuracy: baselineProbeCorrect / n,
    postInterventionProbeAccuracy: postProbeCorrect / n,
    baselineModelAccuracy: baselineModelCorrect / n,
    postInterventionModelAccuracy: postModelCorrect / n,
    causalModelFlippingRate: modelFlipped / n,
    probePredictionChangeRate: probeFlipped / n,
  };
}
