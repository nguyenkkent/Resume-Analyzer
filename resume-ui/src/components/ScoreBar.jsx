// Horizontal bar that visualizes a similarity score [0...1].
export default function ScoreBar({ label, value }) {
  // Convert to percent and bound them to 0 and 1 to prevent invalid values
  const percent = Math.max(0, Math.min(1, Number(value || 0))) * 100;
  return (
    <div className="flex items-center gap-3">
      <div className="w-48 truncate text-sm text-gray-600">{label}</div>
      <div className="h-2 flex-1 rounded-full bg-gray-200">
        <div
          className="h-2 rounded-full bg-blue-500"
          style={{ width: `${Math.round(percent)}%` }}
        />
      </div>
      <div className="w-12 text-right text-sm text-gray-700">
        {/* Cap to two decimal spaces */}
        {(value ?? 0).toFixed(2)}
      </div>
    </div>
  );
}
