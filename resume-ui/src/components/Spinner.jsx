// A tiny spinner used while the agent works.
export default function Spinner({ label = "Working…" }) {
  return (
    <div className="inline-flex items-center gap-2 rounded-full bg-black px-4 py-2 text-white">
      <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24">
        <path
          fill="currentColor"
          d="M12 2a1 1 0 1 1 0 2a8 8 0 1 0 8 8a1 1 0 1 1 2 0a10 10 0 1 1-10-10"
        />
      </svg>
      <span>{label}</span>
    </div>
  );
}
