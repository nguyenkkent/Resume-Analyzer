import { useEffect, useRef, useState } from "react";
import { Toaster, toast } from "react-hot-toast";
import Spinner from "./components/Spinner";
import MessageBubble from "./components/MessageBubble";
import JobCard from "./components/JobCard";
import ScoreBar from "./components/ScoreBar";

/**
 * This component renders a full-height chat layout:
 *  - Sticky header (title + loading spinner)
 *  - Resume input box (collapsible) where the user pastes raw text
 *  - Scrollable chat history
 *  - User text input at the bottom
 *  - Results section (jobs + similarity scores) below the chat
 */

export default function App() {
  const [resumeText, setResumeText] = useState("");
  const [isResumeDrawerOpen, setIsResumeDrawerOpen] = useState(true);

  const [outgoingMessage, setOutgoingMessage] = useState("");
  const [isWorking, setIsWorking] = useState(false);

  // chatHistory: array of { role: "user" | "agent", text: string }
  const [chatHistory, setChatHistory] = useState([]);

  // lastAgentResult mirrors the server response from /agent/chat
  const [lastAgentResult, setLastAgentResult] = useState(null);

  // Ref so we can auto-scroll chat to the latest message
  const chatScrollRef = useRef(null);

  // Whenever messages change, scroll to bottom.
  useEffect(() => {
    if (!chatScrollRef.current) return;
    chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
  }, [chatHistory, isWorking]);


  async function sendToAgent() {
    const trimmed = outgoingMessage.trim();
    if (!trimmed) return;

    // push the user message into the chat history UI
    setChatHistory((prev) => [...prev, { role: "user", text: trimmed }]);

    setIsWorking(true);
    const toastId = toast.loading("Contacting agent…");

    try {
      const response = await fetch("/agent/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: trimmed,
          resume_text: resumeText || undefined,
          limit: 5,
        }),
      });

      // Response might not always be JSON
      let responseData = {};
      try {
        responseData = await response.json();
      } catch {
        // keep default empty object if parsing fails
      }

      if (!response.ok) {
        // If server returned a helpful message, show it.
        const serverMessage =
          responseData?.detail || `Request failed (${response.status})`;
        toast.error(serverMessage, { id: toastId });
        setChatHistory((prev) => [
          ...prev,
          { role: "agent", text: `❌ ${serverMessage}` },
        ]);
        return;
      }

      // On success push agent reply, store structured result
      toast.success("Got results!", { id: toastId });
      setChatHistory((prev) => [
        ...prev,
        { role: "agent", text: String(responseData.reply || "Done.") },
      ]);
      setLastAgentResult(responseData);
    } catch (error) {
      // Network or unexpected error
      toast.error(error?.message || "Network error", { id: toastId });
      setChatHistory((prev) => [
        ...prev,
        { role: "agent", text: `❌ ${error?.message || "Network error"}` },
      ]);
    } finally {
      setOutgoingMessage("");
      setIsWorking(false);
    }
  }

  // Copy any object as JSON
  function copyAsJson(obj, toastLabel = "Copied JSON") {
    const pretty = JSON.stringify(obj, null, 2);
    navigator.clipboard
      .writeText(pretty)
      .then(() => toast.success(toastLabel))
      .catch(() => toast.error("Clipboard not available"));
  }

  const hasJobs = !!lastAgentResult?.jobs?.length;
  const hasScores = !!lastAgentResult?.scores?.length;

  return (
    <div className="flex h-dvh flex-col font-sans">
      <Toaster position="top-right" />

      {/* Header */}
      <header className="sticky top-0 z-10 border-b border-gray-200 bg-white/80 px-5 py-3 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <h1 className="text-xl font-bold tracking-tight">
            Resume Analyzer and Job Matcher Agent
          </h1>
          {isWorking ? <Spinner label="Analyzing…" /> : null}
        </div>
      </header>

      {/* Main content area */}
      <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col px-5">
        {/* Resume input */}
        <section className="my-4">
          <button
            onClick={() => setIsResumeDrawerOpen((v) => !v)}
            className="inline-flex items-center gap-2 rounded-xl border border-gray-300 bg-white px-3 py-2 text-sm hover:bg-gray-50"
          >
            {isResumeDrawerOpen ? "Hide" : "Show"} Resume
          </button>

          {isResumeDrawerOpen ? (
            <div className="mt-3 rounded-2xl border border-gray-200 bg-white p-3">
              <label className="mb-2 block text-sm font-semibold text-gray-800">
                Your Resume (paste raw text)
              </label>
              <textarea
                value={resumeText}
                onChange={(e) => setResumeText(e.target.value)}
                rows={8}
                placeholder="Paste resume text here…"
                className="h-48 w-full resize-y rounded-xl border border-gray-200 p-3 outline-none ring-0 focus:border-gray-300"
              />
              <div className="mt-2 text-xs text-gray-500">
                Tip: You can keep this open while chatting, or hide it for more
                space.
              </div>
            </div>
          ) : null}
        </section>

        {/* Chat area */}
        <section
          ref={chatScrollRef}
          className="grid flex-1 gap-3 overflow-y-auto rounded-2xl border border-gray-200 bg-white p-3"
        >
          {chatHistory.length === 0 ? (
            <div className="mx-auto my-10 max-w-lg text-center text-gray-500">
              <div className="text-sm">
                Paste your resume above, then enter a search prompt. For example:
              </div>
              <div className="mt-1 rounded-lg border border-dashed border-gray-300 bg-gray-50 px-3 py-2 text-sm">
                “Find machine learning engineer roles in SF (hybrid)”
              </div>
            </div>
          ) : (
            chatHistory.map((message, index) => (
              <MessageBubble key={index} role={message.role} text={message.text} />
            ))
          )}
          {isWorking && (
            <div className="justify-self-start rounded-xl border border-gray-100 bg-gray-50 p-3 text-sm text-gray-700">
              Agent is thinking…
            </div>
          )}
        </section>

        {/* User input area */}
        <section className="sticky bottom-0 mt-3 bg-white py-3">
          <div className="flex items-center gap-2">
            <input
              value={outgoingMessage}
              onChange={(e) => setOutgoingMessage(e.target.value)}
              onKeyDown={(e) =>
                e.key === "Enter" && !e.shiftKey ? sendToAgent() : null
              }
              placeholder='Try: "Find machine learning engineer roles in SF (hybrid)"'
              className="flex-1 rounded-xl border border-gray-300 px-3 py-3 outline-none focus:border-gray-400"
            />
            <button
              onClick={sendToAgent}
              disabled={isWorking}
              className={[
                "min-w-[96px] rounded-xl border px-4 py-3 text-white",
                isWorking
                  ? "cursor-not-allowed border-gray-700 bg-gray-700"
                  : "border-black bg-black hover:bg-gray-900",
              ].join(" ")}
            >
              {isWorking ? "Working…" : "Send"}
            </button>
          </div>
        </section>

        {/* Agent response */}
        {hasJobs || hasScores ? (
          <section className="my-5 grid gap-5">
            {hasJobs ? (
              <div>
                <div className="mb-2 flex items-center gap-2">
                  <h2 className="text-lg font-semibold">Jobs</h2>
                  <span className="rounded-full border border-indigo-200 bg-indigo-50 px-2 py-0.5 text-xs text-indigo-800">
                    {lastAgentResult.jobs.length}
                  </span>
                  <button
                    onClick={() => copyAsJson(lastAgentResult.jobs, "Copied jobs")}
                    className="ml-auto rounded-lg border border-gray-200 bg-gray-50 px-3 py-1.5 text-xs hover:bg-gray-100"
                  >
                    Copy Jobs JSON
                  </button>
                </div>
                <div className="grid gap-3 md:grid-cols-2">
                  {lastAgentResult.jobs.map((job) => (
                    <JobCard key={job.id} job={job} />
                  ))}
                </div>
              </div>
            ) : null}

            {hasScores ? (
              <div>
                <div className="mb-2 flex items-center gap-2">
                  <h2 className="text-lg font-semibold">Similarity Scores</h2>
                  <button
                    onClick={() =>
                      copyAsJson(lastAgentResult.scores, "Copied scores")
                    }
                    className="ml-auto rounded-lg border border-gray-200 bg-gray-50 px-3 py-1.5 text-xs hover:bg-gray-100"
                  >
                    Copy Scores
                  </button>
                </div>
                <div className="grid gap-3">
                  {lastAgentResult.scores.map((s) => (
                    <ScoreBar key={s.job_id} label={s.job_id} value={s.similarity} />
                  ))}
                </div>
              </div>
            ) : null}
          </section>
        ) : null}
      </main>
    </div>
  );
}
