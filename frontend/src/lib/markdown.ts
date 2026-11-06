import { marked } from 'marked';
import katex from 'katex';

/**
 * Custom marked extension or preprocessor for KaTeX math.
 * Replaces $$math$$ with display math, and $math$ with inline math.
 */
export function renderMarkdown(content: string): string {
  // Replace $$ display math $$
  let processed = content.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => {
    try {
      return katex.renderToString(tex.trim(), { displayMode: true, throwOnError: false });
    } catch {
      return `<div class="katex-error">${tex}</div>`;
    }
  });

  // Replace $ inline math $ (excluding literal \$)
  processed = processed.replace(/(?<!\\)\$([^\$\n]+?)(?<!\\)\$/g, (_, tex) => {
    try {
      return katex.renderToString(tex.trim(), { displayMode: false, throwOnError: false });
    } catch {
      return `<span class="katex-error">${tex}</span>`;
    }
  });

  return marked.parse(processed, {
    gfm: true,
    breaks: false,
  }) as string;
}
