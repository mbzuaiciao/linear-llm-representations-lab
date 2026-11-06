# Linear LLM Representations Lab — Interactive Research Frontend

An Apple-inspired, static-first interactive research textbook and empirical artifact explorer for the [Linear LLM Representations Lab](../).

## Architecture & Design Principles

- **Static-first**: Built with [Astro](https://astro.build/) for static HTML generation (`output: 'static'`). Zero backend API, zero databases, zero authentication, and zero expensive model downloads required for reading or interactive exploration.
- **Apple-inspired visual language**: Restrained typography using the native Apple system font stack (`SF Pro Text`, `SF Pro Display`, `SF Mono`), generous whitespace, subtle translucent navigation surfaces (`backdrop-filter: blur(20px)`), delicate borders, responsive layouts, accessible contrast, and smooth light/dark theme switching.
- **Epistemic rigor**: Never collapses distinct standards of evidence. Surfaces the central epistemic hierarchy:
  $$\text{Decodability} \longrightarrow \text{Predictability} \longrightarrow \text{Low-Complexity Predictability} \longrightarrow \text{OOD Generalization} \longrightarrow \text{Causal Linear Structure}$$
- **Faithful artifact presentation**: Directly displays committed machine-readable results (`results/capstone.json`, `results/capstone.md`). Absent figures or unrun sweeps receive clear, honest fallback indicators; results are never fabricated or simulated.
- **Deterministic client-side explainers**: Features two pure-TypeScript mathematical widgets illustrating (1) Global Affine Fit vs. Local First-Order Tangent (Tutorial 01 & 08), and (2) Linear Probe Decodability vs. Causal Intervention (Tutorial 03).

---

## Directory Structure

```text
frontend/
├── package.json              # Scripts and dependencies
├── astro.config.mjs          # Astro configuration (static output, Tailwind plugin)
├── tsconfig.json             # TypeScript configuration
├── scripts/
│   └── sync-assets.js        # Build-time synchronization from ../figures to public/figures
├── public/
│   ├── favicon.svg           # Minimal lab icon
│   └── figures/              # Synced experiment figures
├── src/
│   ├── components/
│   │   ├── Header.astro              # Translucent header with search, progress, theme toggle
│   │   ├── Footer.astro              # Repository map, CLI hints, licensing
│   │   ├── ThemeToggle.astro         # Zero-FOUC light/dark/system theme switcher
│   │   ├── SearchModal.astro         # Accessible Cmd+K search dialog
│   │   ├── ReadingProgress.astro     # localStorage-persisted reading tracker
│   │   ├── EpistemicLadder.astro     # 5-tier evidence hierarchy component
│   │   ├── Explainers/
│   │   │   ├── LocalVsGlobalExplainer.astro  # Interactive Jacobian vs affine tool
│   │   │   └── ProbeVsCausalExplainer.astro  # Interactive probe vs causality tool
│   │   └── Results/
│   │       ├── CapstoneViewer.astro          # Comprehensive capstone artifact viewer
│   │       └── ExperimentCard.astro          # Component sweep cards with fallback states
│   ├── layouts/
│   │   └── BaseLayout.astro          # Root HTML layout with SEO meta & theme initializer
│   ├── lib/
│   │   ├── content.ts                # Parsers for all 14 tutorials and 13 notes
│   │   ├── results.ts                # Parser for results JSON/MD artifacts
│   │   ├── markdown.ts               # KaTeX-enabled Markdown renderer
│   │   ├── math-explainer.ts         # Pure mathematical routines for interactive widgets
│   │   └── search-index.ts           # Static searchable index builder
│   ├── pages/
│   │   ├── index.astro               # Home / Overview & Epistemic framework
│   │   ├── tutorials/
│   │   │   ├── index.astro           # 14 tutorials index
│   │   │   └── [id].astro            # Dynamic tutorial detail (01 through 14)
│   │   ├── notes/
│   │   │   ├── index.astro           # 13 conceptual notes index
│   │   │   └── [slug].astro          # Dynamic note detail
│   │   ├── results/
│   │   │   └── index.astro           # Committed experiments explorer
│   │   ├── explainers/
│   │   │   └── index.astro           # Dedicated interactive explainers page
│   │   └── search-index.json.ts      # Static search API endpoint
│   └── styles/
│       └── global.css                # Typography, color tokens, KaTeX styles
├── tests/
│   ├── content.test.ts               # Tests for 14 tutorials & 13 notes parsing
│   ├── results.test.ts               # Tests for capstone & experiment artifacts
│   └── explainers.test.ts            # Mathematical verification of explainers
└── vitest.config.ts                  # Vitest runner configuration
```

---

## Local Development & Setup

### Prerequisites
- Node.js 18+ (tested with Node v26)
- npm 9+

### Installation

```bash
cd frontend
npm install
```

### Development Server

```bash
npm run dev
```

Starts the local development server at `http://localhost:4321`. Automatically triggers `npm run sync:assets` before starting to ensure any figures in `figures/` are available.

### Type & Diagnostics Checking

```bash
npm run check
```

Runs `astro check` to validate all Astro components, TypeScript types, and imports.

### Unit Tests

```bash
npm test
```

Runs the Vitest test suite (`tests/content.test.ts`, `tests/results.test.ts`, `tests/explainers.test.ts`).

### Production Build & Preview

```bash
npm run build
npm run preview
```

Generates 32 static HTML pages in `frontend/dist/` ready for hosting on any static provider (GitHub Pages, Cloudflare Pages, Netlify, Vercel, S3).

---

## Static Artifact Refresh Behavior

The frontend is intentionally designed as an offline static consumer of the Python lab's artifacts:

1. **Python Experiments Run Independently**:
   Researchers run experiments from the repository root using `uv`:
   ```bash
   uv run python experiments/run_capstone.py --transformer --samples 60
   uv run python experiments/run_rank_sweep.py
   uv run python experiments/run_local_linearity.py
   ```
   These commands output machine-readable JSON files to `results/` and graphical plots to `figures/`.

2. **Build-Time Ingestion**:
   When building or developing the frontend (`npm run build` or `npm run dev`), the `prebuild`/`predev` hook executes `node scripts/sync-assets.js`. This automatically copies any generated figures from `../figures/` to `frontend/public/figures/`.
   
   The content loaders in `src/lib/results.ts` read directly from `results/*.json` and `results/capstone.md`.

3. **No Code Modification Required**:
   New experimental outputs become immediately visible in the web app upon re-running `npm run build` without modifying a single line of scientific Python code.

---

## Known Limitations & Epistemic Boundaries

1. **Static Build vs. Live Python Kernel**:
   The frontend does not host a live Python runtime or server-side Jupyter kernel. Exploring new seeds, prompts, or models requires running the Python CLI commands in your terminal and rebuilding the static site.
2. **Pedagogical Explainers are Not Empirical Evidence**:
   The interactive widgets on the Explainers page are deterministic mathematical illustrations of Taylor theorem convergence and linear readout properties. They are not neural network activations and are explicitly labeled as pedagogical demonstrations.
3. **Capstone Model Capacity**:
   The committed capstone experiment runs on `sshleifer/tiny-gpt2` with hidden width $d=2$. An affine map across $d=2$ has only 6 parameters, meaning Rank-1 captures nearly full capacity. Caveats regarding prompt template leakage and logit calibration are highlighted directly alongside the quantitative table.
