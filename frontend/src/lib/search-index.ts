import { getAllTutorials, getAllNotes } from './content';
import { getAllExperiments } from './results';

export interface SearchEntry {
  id: string;
  title: string;
  category: 'Tutorial' | 'Note' | 'Experiment' | 'Explainer';
  url: string;
  badge: string;
  summary: string;
}

export function buildSearchIndex(): SearchEntry[] {
  const tutorials = getAllTutorials();
  const notes = getAllNotes();
  const experiments = getAllExperiments();

  const entries: SearchEntry[] = [];

  // Explainers
  entries.push({
    id: 'explainer-local-global',
    title: 'Interactive: Global Affine vs Local Linearization',
    category: 'Explainer',
    url: '/explainers#local-global',
    badge: 'Interactive',
    summary: 'Compare global affine fit with first-order Taylor tangent and relative error across perturbation radii.',
  });

  entries.push({
    id: 'explainer-probe-causal',
    title: 'Interactive: Linear Probe Decodability vs Causal Intervention',
    category: 'Explainer',
    url: '/explainers#probe-causal',
    badge: 'Interactive',
    summary: 'Test how high linear decodability can co-exist with zero downstream causal effect.',
  });

  // Tutorials
  for (const t of tutorials) {
    entries.push({
      id: `tutorial-${t.id}`,
      title: `Tutorial ${t.id}: ${t.title}`,
      category: 'Tutorial',
      url: `/tutorials/${t.id}`,
      badge: t.centralDistinction,
      summary: `${t.intuition} ${t.interpretation}`,
    });
  }

  // Notes
  for (const n of notes) {
    entries.push({
      id: `note-${n.slug}`,
      title: n.title,
      category: 'Note',
      url: `/notes/${n.slug}`,
      badge: `${n.readingTimeMinutes} min read`,
      summary: n.rawContent.slice(0, 200).replace(/#+\s+/g, '').replace(/\n+/g, ' '),
    });
  }

  // Experiments
  for (const e of experiments) {
    entries.push({
      id: `exp-${e.id}`,
      title: e.title,
      category: 'Experiment',
      url: `/results#${e.id}`,
      badge: e.modelType,
      summary: `${e.subtitle}. Caveat: ${e.primaryCaveat}`,
    });
  }

  return entries;
}
