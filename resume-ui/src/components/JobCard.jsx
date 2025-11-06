// Displays a single job result from the agent.
export default function JobCard({ job }) {
  return (
    <div className="rounded-xl border border-gray-200 p-4 shadow-sm">
      <div className="flex items-baseline justify-between gap-3">
        <div>
          <div className="text-base font-semibold text-gray-900">
            {job.title || "Job"}
          </div>
          <div className="text-sm text-gray-600">
            {job.company || "Unknown company"}
          </div>
        </div>
        {job.url ? (
          <a
            href={job.url}
            target="_blank"
            rel="noreferrer"
            className="text-sm text-blue-600 hover:underline"
          >
            URL
          </a>
        ) : null}
      </div>
      {job.summary ? (
        <p className="mt-2 text-[15px] text-gray-800">{job.summary}</p>
      ) : null}
    </div>
  );
}
