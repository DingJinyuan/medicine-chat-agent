import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// LLM output occasionally mixes literal <br> tags into markdown tables; we
// normalize those to newlines rather than enabling raw HTML rendering
// (rehype-raw), which would let injected content render arbitrary HTML.
function normalize(content: string): string {
  return content.replace(/<br\s*\/?>/gi, "\n");
}

export function MarkdownAnswer({ content }: { content: string }) {
  return (
    <div className="prose prose-sm dark:prose-invert max-w-none prose-p:leading-relaxed prose-p:my-2 first:prose-p:mt-0 last:prose-p:mb-0 prose-table:text-xs prose-headings:mt-3 prose-headings:mb-1.5">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{normalize(content)}</ReactMarkdown>
    </div>
  );
}
