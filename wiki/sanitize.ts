/**
 * sanitize-html.ts
 * Quick fixer para snapshots de SingleFile que hacen tropezar a parsers/LLMs (como v0).
 * - Crea <head> si no existe y mueve ahí todo lo pre-<body>.
 * - Opcionalmente elimina <style> (reduce mucho el tamaño) y data:URLs.
 * - Quita iframes ocultos y favicons base64.
 * - Asegura cierre de </body></html>.
 *
 * Uso:
 *   npx ts-node sanitize-html.ts <input.html> [--out output.html] [--keep-css] [--keep-iframes]
 */
import fs from 'node:fs';
import path from 'node:path';

function readFile(p: string): string {
  return fs.readFileSync(p, 'utf8');
}
function writeFile(p: string, content: string): void {
  fs.writeFileSync(p, content, 'utf8');
}

function ensureHead(html: string): string {
  const htmlOpen = html.match(/<html\b[^>]*>/i);
  if (!htmlOpen) return html;
  const startIdx = (htmlOpen.index ?? 0) + htmlOpen[0].length;
  const afterHtml = html.slice(startIdx);
  const bodyMatch = afterHtml.match(/<body\b[^>]*>/i);
  const beforeBody = bodyMatch ? afterHtml.slice(0, bodyMatch.index) : afterHtml;
  const hasHead = /<head\b/i.test(beforeBody);
  if (hasHead) return html;
  return (
    html.slice(0, startIdx) +
    '<head>' +
    beforeBody +
    '</head>' +
    afterHtml.slice(beforeBody.length)
  );
}

function stripStyles(html: string): string {
  return html.replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, '');
}
function removeBase64Favicons(html: string): string {
  return html.replace(
    /<link\b[^>]*rel=["']?(?:shortcut\s+icon|icon)["']?[^>]*href=["']?data:[^>"']+["']?[^>]*>/gi,
    ''
  );
}
function neutralizeDataUrlsInCss(html: string): string {
  return html.replace(/url\((["']?)data:[^)"']+\1\)/gi, 'none');
}
function stripIframes(html: string): string {
  return html.replace(/<iframe\b[\s\S]*?<\/iframe>/gi, '');
}
function ensureClosingTags(html: string): string {
  let out = html;
  if (/<body\b/i.test(out) && !/<\/body>/i.test(out)) out += '</body>';
  if (!/<\/html>/i.test(out)) out += '</html>';
  return out;
}
function fixAttributeQuotes(html: string): string {
  let out = html.replace(/<html\b([^>]*\blang=)([^\s>"']+)/i, '<html $1"$2"');
  out = out.replace(/<html\b([^>]*\bstyle=)([^\s>"']+)/i, '<html $1"$2"');
  return out;
}

function sanitize(html: string, opts: { keepCss: boolean; keepIframes: boolean }): string {
  let out = html;
  out = ensureHead(out);
  out = removeBase64Favicons(out);
  out = neutralizeDataUrlsInCss(out);
  if (!opts.keepCss) out = stripStyles(out);
  if (!opts.keepIframes) out = stripIframes(out);
  out = ensureClosingTags(out);
  out = fixAttributeQuotes(out);
  return out;
}

function main(): void {
  const args = process.argv.slice(2);
  if (args.length === 0) {
    console.error(
      'Uso: npx ts-node sanitize-html.ts <input.html> [--out output.html] [--keep-css] [--keep-iframes]'
    );
    process.exit(1);
  }
  const inputPath = args[0];
  const outFlagIdx = args.indexOf('--out');
  const outputPath =
    outFlagIdx !== -1 && args[outFlagIdx + 1]
      ? args[outFlagIdx + 1]
      : path.join(
          path.dirname(inputPath),
          path.basename(inputPath).replace(/\.html?$/i, '') + '.sanitized.html'
        );
  const keepCss = args.includes('--keep-css');
  const keepIframes = args.includes('--keep-iframes');
  const raw = readFile(inputPath);
  const cleaned = sanitize(raw, { keepCss, keepIframes });
  writeFile(outputPath, cleaned);
  console.log(`Wrote: ${outputPath}`);
}
main();
