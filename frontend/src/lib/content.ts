import fs from 'node:fs';
import path from 'node:path';

export interface TutorialItem {
  id: string; // "01", "02", ...
  number: number;
  slug: string;
  title: string;
  centralDistinction: string;
  intuition: string;
  mathematics: string;
  experiment: string;
  interpretation: string;
  sourceFile: string;
  companionNoteSlug?: string;
  cliCommand: string;
  runnerCode: string;
  lessonCode: string;
}

export interface NoteItem {
  slug: string;
  title: string;
  rawContent: string;
  sourceFile: string;
  readingTimeMinutes: number;
  relatedTutorialIds: string[];
}

function findRepoRoot(): string {
  // Check if cwd is repo root or frontend
  const cwd = process.cwd();
  if (fs.existsSync(path.join(cwd, 'tutorials')) && fs.existsSync(path.join(cwd, 'notes'))) {
    return cwd;
  }
  const parent = path.resolve(cwd, '..');
  if (fs.existsSync(path.join(parent, 'tutorials')) && fs.existsSync(path.join(parent, 'notes'))) {
    return parent;
  }
  // Fallback
  return path.resolve(cwd, '..');
}

const TUTORIAL_METADATA: Record<number, { slug: string; distinction: string; companionNote: string }> = {
  1: { slug: 'linear-maps', distinction: 'global fit versus local Jacobian', companionNote: 'local_linearization' },
  2: { slug: 'residual-stream', distinction: 'additive updates versus global linearity', companionNote: 'residual_stream' },
  3: { slug: 'linear-probes', distinction: 'decodability versus causal computation', companionNote: 'linear_probes_vs_computation' },
  4: { slug: 'layer-affine-maps', distinction: 'learned map versus trivial baselines', companionNote: 'representation_manifolds' },
  5: { slug: 'high-dimensional-trap', distinction: 'predictability versus compressibility', companionNote: 'failure_modes' },
  6: { slug: 'token-trajectories', distinction: 'full geometry versus PCA appearance', companionNote: 'trajectory_straightening' },
  7: { slug: 'concept-directions', distinction: 'decoding versus steering versus mechanism', companionNote: 'concept_directions' },
  8: { slug: 'local-linearization', distinction: 'infinitesimal truth versus useful radius', companionNote: 'local_linearization' },
  9: { slug: 'activation-manifolds', distinction: 'natural manifold versus ambient space', companionNote: 'representation_manifolds' },
  10: { slug: 'global-vs-local', distinction: 'global affine versus piecewise affine', companionNote: 'linearity_taxonomy' },
  11: { slug: 'latent-dynamics', distinction: 'descriptive spectra versus dynamics', companionNote: 'latent_dynamical_systems' },
  12: { slug: 'koopman-intuition', distinction: 'nonlinear states versus simple coordinates', companionNote: 'koopman_intuition' },
  13: { slug: 'cross-model-alignment', distinction: 'reconstruction versus transferable behavior', companionNote: 'cross_model_alignment' },
  14: { slug: 'capstone', distinction: 'interpolation versus structural generalization', companionNote: 'failure_modes' },
};

export function getAllTutorials(): TutorialItem[] {
  const root = findRepoRoot();
  const tutorialsDir = path.join(root, 'tutorials');
  const sharedPyPath = path.join(tutorialsDir, '_shared.py');
  const sharedPy = fs.existsSync(sharedPyPath) ? fs.readFileSync(sharedPyPath, 'utf-8') : '';

  const tutorials: TutorialItem[] = [];

  for (let num = 1; num <= 14; num++) {
    const id = num.toString().padStart(2, '0');
    const meta = TUTORIAL_METADATA[num];
    const fileName = `${id}_${meta.slug.replace(/-/g, '_')}.py`;
    const fullPath = path.join(tutorialsDir, fileName);

    let content = '';
    if (fs.existsSync(fullPath)) {
      content = fs.readFileSync(fullPath, 'utf-8');
    }

    // Parse docstring
    // Example: """Tutorial 01 — Linear maps. Intuition: superposition is exact for Ax. Mathematics: affine maps add b and Jacobians give local slopes. Experiment: fit sin(x) globally and locally. Interpretation: local success does not make sin globally linear."""
    const docMatch = content.match(/"""([\s\S]*?)"""/);
    const doc = docMatch ? docMatch[1].trim() : '';

    const titleMatch = doc.match(/Tutorial \d+\s*[—–-]\s*([^.]+)\./i);
    const title = titleMatch ? titleMatch[1].trim() : `Tutorial ${id}`;

    const intuitionMatch = doc.match(/Intuition:\s*([^.]+(?:\.[^.]+)?(?=\s*Mathematics:|$))/i);
    const mathMatch = doc.match(/Mathematics:\s*([^.]+(?:\.[^.]+)?(?=\s*Experiment:|$))/i);
    const expMatch = doc.match(/Experiment:\s*([^.]+(?:\.[^.]+)?(?=\s*Interpretation:|$))/i);
    const interpMatch = doc.match(/Interpretation:\s*([^]+)$/i);

    // Extract lesson_XX from _shared.py
    const funcName = `lesson_${id}`;
    let lessonCode = '';
    const funcPattern = new RegExp(`def ${funcName}\\([\\s\\S]*?(?=\\n(?:def |if __name__|$))`);
    const matchFunc = sharedPy.match(funcPattern);
    if (matchFunc) {
      lessonCode = matchFunc[0].trim();
    }

    tutorials.push({
      id,
      number: num,
      slug: meta.slug,
      title,
      centralDistinction: meta.distinction,
      intuition: intuitionMatch ? intuitionMatch[1].trim() : '',
      mathematics: mathMatch ? mathMatch[1].trim() : '',
      experiment: expMatch ? expMatch[1].trim() : '',
      interpretation: interpMatch ? interpMatch[1].trim() : '',
      sourceFile: `tutorials/${fileName}`,
      companionNoteSlug: meta.companionNote,
      cliCommand: `uv run python tutorials/${fileName}${num === 2 ? ' --synthetic' : ''}`,
      runnerCode: content.trim(),
      lessonCode,
    });
  }

  return tutorials;
}

export function getTutorialById(id: string): TutorialItem | undefined {
  const normId = id.padStart(2, '0');
  return getAllTutorials().find((t) => t.id === normId);
}

export function getAllNotes(): NoteItem[] {
  const root = findRepoRoot();
  const notesDir = path.join(root, 'notes');
  if (!fs.existsSync(notesDir)) {
    return [];
  }

  const files = fs.readdirSync(notesDir).filter((f) => f.endsWith('.md')).sort();

  return files.map((file) => {
    const slug = file.replace(/\.md$/, '');
    const fullPath = path.join(notesDir, file);
    const rawContent = fs.readFileSync(fullPath, 'utf-8');

    // Extract title from first # Heading
    const titleMatch = rawContent.match(/^#\s+(.+)$/m);
    const title = titleMatch ? titleMatch[1].trim() : slug.replace(/_/g, ' ');

    // Calculate approximate reading time (words / 180 wpm)
    const wordCount = rawContent.split(/\s+/).length;
    const readingTimeMinutes = Math.max(1, Math.round(wordCount / 180));

    // Find related tutorials
    const relatedTutorialIds = Object.entries(TUTORIAL_METADATA)
      .filter(([_, meta]) => meta.companionNote === slug)
      .map(([num]) => num.padStart(2, '0'));

    return {
      slug,
      title,
      rawContent,
      sourceFile: `notes/${file}`,
      readingTimeMinutes,
      relatedTutorialIds,
    };
  });
}

export function getNoteBySlug(slug: string): NoteItem | undefined {
  return getAllNotes().find((n) => n.slug === slug);
}
