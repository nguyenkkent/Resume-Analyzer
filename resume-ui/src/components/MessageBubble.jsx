// Renders a chat bubble
export default function MessageBubble({ role, text }) {
  // "user" | "agent"
  const isUser = role === "user";
  return (
    <div
      className={[
        "max-w-[80%] rounded-2xl border p-3 whitespace-pre-wrap",
        isUser
          ? "self-end justify-self-end border-blue-100 bg-blue-50"
          : "self-start justify-self-start border-gray-100 bg-gray-50",
      ].join(" ")}
    >
      <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-gray-500">
        {isUser ? "You" : "Agent"}
      </div>
      <div className="text-[15px] leading-relaxed text-gray-900">{text}</div>
    </div>
  );
}
