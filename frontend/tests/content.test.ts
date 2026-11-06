import { describe, expect, it } from 'vitest';
import { getAllTutorials, getTutorialById, getAllNotes, getNoteBySlug } from '../src/lib/content';

describe('Content Loader', () => {
  it('loads all 14 tutorials with required structure and non-empty sections', () => {
    const tutorials = getAllTutorials();
    expect(tutorials).toHaveLength(14);

    for (let num = 1; num <= 14; num++) {
      const id = num.toString().padStart(2, '0');
      const tutorial = getTutorialById(id);
      expect(tutorial).toBeDefined();
      if (!tutorial) continue;

      expect(tutorial.id).toBe(id);
      expect(tutorial.number).toBe(num);
      expect(tutorial.title.length).toBeGreaterThan(0);
      expect(tutorial.centralDistinction.length).toBeGreaterThan(0);
      expect(tutorial.intuition.length).toBeGreaterThan(0);
      expect(tutorial.mathematics.length).toBeGreaterThan(0);
      expect(tutorial.experiment.length).toBeGreaterThan(0);
      expect(tutorial.interpretation.length).toBeGreaterThan(0);
      expect(tutorial.cliCommand).toContain('uv run python tutorials/');
      expect(tutorial.sourceFile).toContain(`tutorials/${id}_`);
      expect(tutorial.runnerCode.length).toBeGreaterThan(0);
      expect(tutorial.lessonCode.length).toBeGreaterThan(0);
      expect(tutorial.companionNoteSlug).toBeDefined();
    }
  });

  it('loads all 13 conceptual notes with title, content and reading time', () => {
    const notes = getAllNotes();
    expect(notes.length).toBe(13);

    const expectedSlugs = [
      'concept_directions',
      'cross_model_alignment',
      'failure_modes',
      'glossary',
      'koopman_intuition',
      'latent_dynamical_systems',
      'linear_probes_vs_computation',
      'linearity_taxonomy',
      'literature_map',
      'local_linearization',
      'representation_manifolds',
      'residual_stream',
      'trajectory_straightening',
    ];

    for (const slug of expectedSlugs) {
      const note = getNoteBySlug(slug);
      expect(note).toBeDefined();
      if (!note) continue;

      expect(note.slug).toBe(slug);
      expect(note.title.length).toBeGreaterThan(0);
      expect(note.rawContent.length).toBeGreaterThan(0);
      expect(note.readingTimeMinutes).toBeGreaterThanOrEqual(1);
      expect(note.sourceFile).toBe(`notes/${slug}.md`);
    }
  });

  it('correctly maps companion notes between tutorials and notes', () => {
    const tutorial01 = getTutorialById('01');
    expect(tutorial01?.companionNoteSlug).toBe('local_linearization');
    const note = getNoteBySlug('local_linearization');
    expect(note).toBeDefined();
    expect(note?.relatedTutorialIds).toContain('01');
    expect(note?.relatedTutorialIds).toContain('08');
  });

  it('renders markdown and equations via KaTeX', async () => {
    const { renderMarkdown } = await import('../src/lib/markdown');
    const html = renderMarkdown('An affine map is $y = Ax + b$.');
    expect(html).toContain('katex');
    expect(html).toContain('affine map');
  });
});
