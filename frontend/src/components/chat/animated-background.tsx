export function AnimatedBackground() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <div className="animate-blob-1 absolute -top-32 -left-24 h-96 w-96 rounded-full bg-primary/20 blur-3xl dark:bg-primary/15" />
      <div className="animate-blob-2 absolute top-1/3 -right-32 h-[28rem] w-[28rem] rounded-full bg-sky-400/15 blur-3xl dark:bg-sky-500/10" />
      <div className="animate-blob-1 absolute bottom-0 left-1/4 h-80 w-80 rounded-full bg-emerald-400/10 blur-3xl dark:bg-emerald-500/10" />
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_1px_1px,var(--color-border)_1px,transparent_0)] bg-[length:24px_24px] opacity-[0.15]" />
    </div>
  );
}
