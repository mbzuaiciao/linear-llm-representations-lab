import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '../..');
const figuresSrc = path.join(repoRoot, 'figures');
const figuresDest = path.join(__dirname, '../public/figures');

if (fs.existsSync(figuresSrc)) {
  fs.mkdirSync(figuresDest, { recursive: true });
  const files = fs.readdirSync(figuresSrc);
  for (const file of files) {
    if (file.endsWith('.png') || file.endsWith('.svg') || file.endsWith('.jpg')) {
      fs.copyFileSync(path.join(figuresSrc, file), path.join(figuresDest, file));
      console.log(`Synced figure: ${file}`);
    }
  }
}
