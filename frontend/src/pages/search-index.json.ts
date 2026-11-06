import { buildSearchIndex } from '../lib/search-index';

export function GET() {
  const index = buildSearchIndex();
  return new Response(JSON.stringify(index), {
    headers: {
      'Content-Type': 'application/json',
    },
  });
}
