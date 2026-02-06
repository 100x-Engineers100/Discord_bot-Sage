"use client";

import DashboardLayout from "@/components/DashboardLayout";
import { useEffect, useState, useMemo } from "react";
import { AlertTriangle, TrendingUp, ChevronDown, ChevronUp } from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type TopicEntry = {
  topic: string;
  query_count: number;
  escalation_count: number;
  query_texts: string[];
  is_trending: boolean;
};

// ---------------------------------------------------------------------------
// TopicCard — expandable card for a single topic
// ---------------------------------------------------------------------------

function TopicCard({ topic }: { topic: TopicEntry }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="bg-white/[0.04] rounded-lg border border-white/[0.08] overflow-hidden hover:border-white/[0.12] transition-colors">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3.5 text-left"
      >
        <div className="flex items-center gap-2.5 flex-wrap">
          <span className="text-white text-sm font-medium">{topic.topic}</span>

          {topic.is_trending && (
            <span className="inline-flex items-center gap-1 text-xs bg-blue-500/[0.1] border border-blue-500/[0.2] text-blue-300 px-2 py-0.5 rounded-full">
              <TrendingUp className="w-3 h-3" />
              Trending
            </span>
          )}

          {topic.escalation_count > 0 && (
            <span className="inline-flex items-center gap-1 text-xs bg-amber-500/[0.1] border border-amber-500/[0.2] text-amber-300 px-2 py-0.5 rounded-full">
              <AlertTriangle className="w-3 h-3" />
              {topic.escalation_count} escalation{topic.escalation_count !== 1 ? "s" : ""}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2.5 shrink-0 ml-4">
          <span className="text-xs bg-green-500/[0.1] border border-green-500/[0.2] text-green-300 px-2 py-0.5 rounded-full font-code">
            {topic.query_count} {topic.query_count === 1 ? "query" : "queries"}
          </span>
          {open ? (
            <ChevronUp className="w-4 h-4 text-gray-600" />
          ) : (
            <ChevronDown className="w-4 h-4 text-gray-600" />
          )}
        </div>
      </button>

      {open && (
        <div className="px-4 pb-3.5 border-t border-white/[0.06] pt-3">
          {topic.query_texts.length > 0 ? (
            <div>
              <p className="text-xs text-gray-600 mb-2 uppercase tracking-wider">Student Queries</p>
              <div className="flex flex-wrap gap-1.5">
                {topic.query_texts.map((qt, i) => (
                  <span
                    key={i}
                    className="text-xs bg-white/[0.04] border border-white/[0.08] text-gray-400 px-2.5 py-1 rounded-md"
                  >
                    {qt}
                  </span>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-xs text-gray-600 italic">No query text recorded</p>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function CommonQueriesPage() {
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState<7 | 30>(30);
  const [topics, setTopics] = useState<TopicEntry[]>([]);

  useEffect(() => {
    fetchTopics();
  }, [timeRange]);

  const fetchTopics = async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/cluster-topics?days=${timeRange}`);
      if (!res.ok) throw new Error(`API returned ${res.status}`);
      const data = await res.json();
      setTopics((data.topics as TopicEntry[]) || []);
    } catch (err) {
      console.error("Error fetching topics:", err);
      setTopics([]);
    } finally {
      setLoading(false);
    }
  };

  // ---------------------------------------------------------------------------
  // Derived: split into escalated vs all
  // ---------------------------------------------------------------------------
  const { escalatedTopics, allTopics } = useMemo(() => {
    const escalated = topics
      .filter((t) => t.escalation_count > 0)
      .sort((a, b) => b.escalation_count - a.escalation_count);
    return { escalatedTopics: escalated, allTopics: topics };
  }, [topics]);

  return (
    <DashboardLayout>
      {/* Time Range Selector */}
      <div className="flex justify-end mb-6">
        <div className="flex gap-1 bg-white/[0.04] rounded-lg p-0.5 border border-white/[0.08]">
          <button
            onClick={() => setTimeRange(7)}
            className={`px-5 py-1.5 rounded-md text-sm font-medium transition-all ${
              timeRange === 7
                ? "bg-green-500/[0.12] text-green-300"
                : "text-gray-500 hover:text-gray-300"
            }`}
          >
            7 Days
          </button>
          <button
            onClick={() => setTimeRange(30)}
            className={`px-5 py-1.5 rounded-md text-sm font-medium transition-all ${
              timeRange === 30
                ? "bg-green-500/[0.12] text-green-300"
                : "text-gray-500 hover:text-gray-300"
            }`}
          >
            30 Days
          </button>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="text-sm text-gray-600 animate-pulse">Loading topics...</div>
        </div>
      ) : (
        <>
          {/* Topics with Escalations */}
          {escalatedTopics.length > 0 && (
            <div className="mb-8">
              <div className="flex items-center gap-2 mb-3">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">Escalated Topics</h2>
              </div>
              <div className="flex flex-col gap-2">
                {escalatedTopics.map((topic, i) => (
                  <TopicCard key={`escalated-${i}`} topic={topic} />
                ))}
              </div>
            </div>
          )}

          {/* All Common Topics */}
          <div>
            <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-3">All Topics</h2>
            {allTopics.length > 0 ? (
              <div className="flex flex-col gap-2">
                {allTopics.map((topic, i) => (
                  <TopicCard key={`topic-${i}`} topic={topic} />
                ))}
              </div>
            ) : (
              <div className="bg-white/[0.03] rounded-xl border border-white/[0.08] p-8 text-center">
                <p className="text-gray-600 text-sm">
                  No query data yet. Topics appear once Sage starts receiving queries.
                </p>
              </div>
            )}
          </div>
        </>
      )}
    </DashboardLayout>
  );
}
