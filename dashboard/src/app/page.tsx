"use client";

import DashboardLayout from "@/components/DashboardLayout";
import { useEffect, useState } from "react";
import { supabase, type AnalyticsEvent } from "@/lib/supabase";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { Activity, CheckCircle2, AlertTriangle, TrendingUp } from "lucide-react";

type MetricsData = {
  totalQueries: number;
  gotItCount: number;
  tagCrewCount: number;
  needHelpCount: number;
  continueCount: number;
  selfServiceRate: number;
  explicitSatisfactionRate: number;
  escalationRate: number;
  engagementRate: number;
};

type TrendData = {
  date: string;
  queries: number;
  gotIt: number;
  tagCrew: number;
}[];

export default function Dashboard() {
  const [metrics, setMetrics] = useState<MetricsData>({
    totalQueries: 0,
    gotItCount: 0,
    tagCrewCount: 0,
    needHelpCount: 0,
    continueCount: 0,
    selfServiceRate: 0,
    explicitSatisfactionRate: 0,
    escalationRate: 0,
    engagementRate: 0,
  });
  const [trendData, setTrendData] = useState<TrendData>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<7 | 30>(30);

  useEffect(() => {
    fetchAnalytics();
  }, [timeRange]);

  const fetchAnalytics = async () => {
    setLoading(true);
    setError(null);
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - timeRange);

    try {
      const { data, error } = await supabase
        .from("analytics_events")
        .select("*")
        .gte("created_at", startDate.toISOString())
        .order("created_at", { ascending: true });

      if (error) throw error;

      const events = data as AnalyticsEvent[];

      const totalQueries = events.filter((e) => e.event_type === "query").length;
      const gotItCount = events.filter((e) => e.event_type === "got_it").length;
      const tagCrewCount = events.filter((e) => e.event_type === "tag_crew").length;
      const needHelpCount = events.filter((e) => e.event_type === "need_help").length;
      const continueCount = events.filter((e) => e.event_type === "continue_here").length;

      const selfServiceRate = totalQueries > 0 ? ((totalQueries - tagCrewCount) / totalQueries) * 100 : 0;
      const explicitSatisfactionRate = (gotItCount + tagCrewCount) > 0 ? (gotItCount / (gotItCount + tagCrewCount)) * 100 : 0;
      const escalationRate = totalQueries > 0 ? (tagCrewCount / totalQueries) * 100 : 0;
      const engagementRate = totalQueries > 0 ? ((gotItCount + needHelpCount) / totalQueries) * 100 : 0;

      setMetrics({
        totalQueries,
        gotItCount,
        tagCrewCount,
        needHelpCount,
        continueCount,
        selfServiceRate,
        explicitSatisfactionRate,
        escalationRate,
        engagementRate,
      });

      const trend: TrendData = [];

      for (let i = 0; i < timeRange; i++) {
        const dayStart = new Date(startDate);
        dayStart.setDate(dayStart.getDate() + i);
        const dayEnd = new Date(dayStart);
        dayEnd.setDate(dayEnd.getDate() + 1);

        const dayEvents = events.filter((e) => {
          const eventDate = new Date(e.created_at);
          return eventDate >= dayStart && eventDate < dayEnd;
        });

        trend.push({
          date: dayStart.toLocaleDateString("en-US", { month: "short", day: "numeric" }),
          queries: dayEvents.filter((e) => e.event_type === "query").length,
          gotIt: dayEvents.filter((e) => e.event_type === "got_it").length,
          tagCrew: dayEvents.filter((e) => e.event_type === "tag_crew").length,
        });
      }

      setTrendData(trend);
    } catch (err) {
      console.error("Error fetching analytics:", err);
      setError((err as Error).message || "Failed to fetch analytics");
    } finally {
      setLoading(false);
    }
  };

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

      {error && (
        <div className="bg-red-500/[0.08] border border-red-500/[0.2] rounded-lg px-4 py-3 mb-6">
          <p className="text-red-400 text-sm">Supabase fetch failed: {error}</p>
        </div>
      )}

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="text-sm text-gray-600 animate-pulse">Loading analytics...</div>
        </div>
      ) : (
        <>
          {/* Hero Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            {/* Self-Service Success */}
            <div className="bg-white/[0.04] rounded-xl border border-white/[0.08] p-5 hover:border-green-500/[0.25] transition-colors">
              <div className="flex items-center gap-3 mb-3">
                <div className="p-2.5 bg-green-500/[0.1] rounded-lg">
                  <CheckCircle2 className="w-5 h-5 text-green-400" />
                </div>
                <div>
                  <p className="text-sm text-gray-400">Self-Service Success</p>
                  <p className="text-xs text-gray-600">Without mentor help</p>
                </div>
              </div>
              <div className="text-4xl font-bold text-green-400 font-code mb-1">
                {metrics.selfServiceRate.toFixed(1)}%
              </div>
              <p className="text-xs text-gray-600 font-code">
                {metrics.totalQueries - metrics.tagCrewCount} / {metrics.totalQueries} queries
              </p>
            </div>

            {/* Total Queries */}
            <div className="bg-white/[0.04] rounded-xl border border-white/[0.08] p-5 hover:border-blue-500/[0.25] transition-colors">
              <div className="flex items-center gap-3 mb-3">
                <div className="p-2.5 bg-blue-500/[0.1] rounded-lg">
                  <Activity className="w-5 h-5 text-blue-400" />
                </div>
                <div>
                  <p className="text-sm text-gray-400">Total Queries</p>
                  <p className="text-xs text-gray-600">Bot responses sent</p>
                </div>
              </div>
              <div className="text-4xl font-bold text-white font-code mb-1">{metrics.totalQueries}</div>
              <div className="flex items-center gap-1.5 text-xs text-gray-600">
                <TrendingUp className="w-3.5 h-3.5" />
                <span>Last {timeRange} days</span>
              </div>
            </div>

            {/* Mentor Escalations */}
            <div className="bg-white/[0.04] rounded-xl border border-white/[0.08] p-5 hover:border-amber-500/[0.25] transition-colors">
              <div className="flex items-center gap-3 mb-3">
                <div className="p-2.5 bg-amber-500/[0.1] rounded-lg">
                  <AlertTriangle className="w-5 h-5 text-amber-400" />
                </div>
                <div>
                  <p className="text-sm text-gray-400">Mentor Escalations</p>
                  <p className="text-xs text-gray-600">Students who tagged mentors</p>
                </div>
              </div>
              <div className="flex items-baseline gap-2.5 mb-1">
                <span className="text-4xl font-bold text-white font-code">{metrics.tagCrewCount}</span>
                <span className="text-xl font-semibold text-amber-400 font-code">
                  {metrics.escalationRate.toFixed(1)}%
                </span>
              </div>
              <p className="text-xs text-gray-600">Of total queries</p>
            </div>
          </div>

          {/* Activity Chart */}
          <div className="bg-white/[0.03] rounded-xl border border-white/[0.08] p-6 mb-8">
            <h2 className="text-sm font-semibold text-gray-300 mb-4">Daily Activity</h2>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={trendData}>
                <XAxis
                  dataKey="date"
                  stroke="#374151"
                  tick={{ fill: '#6b7280', fontSize: 11 }}
                  axisLine={{ stroke: '#1f2937' }}
                  tickLine={false}
                  interval={timeRange === 7 ? 0 : 4}
                />
                <YAxis
                  stroke="#374151"
                  tick={{ fill: '#6b7280', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f1117',
                    border: '1px solid rgba(255,255,255,0.08)',
                    borderRadius: '8px',
                    padding: '10px 14px',
                    boxShadow: '0 4px 24px rgba(0,0,0,0.4)',
                  }}
                  labelStyle={{ color: '#9ca3af', fontWeight: '500', marginBottom: '6px', fontSize: '12px' }}
                  itemStyle={{ color: '#fff', fontSize: '13px', fontFamily: 'var(--font-mono)' }}
                />
                <Legend
                  wrapperStyle={{ paddingTop: '16px' }}
                  iconType="circle"
                  iconSize={8}
                />
                <Line type="monotone" dataKey="queries" stroke="#4ade80" strokeWidth={2} dot={{ fill: '#4ade80', r: 3 }} activeDot={{ r: 4 }} name="Total Queries" />
                <Line type="monotone" dataKey="gotIt" stroke="#22d3ee" strokeWidth={2} dot={{ fill: '#22d3ee', r: 3 }} activeDot={{ r: 4 }} name="Got It" />
                <Line type="monotone" dataKey="tagCrew" stroke="#fbbf24" strokeWidth={2} dot={{ fill: '#fbbf24', r: 3 }} activeDot={{ r: 4 }} name="Tagged Crew" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Secondary Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-white/[0.03] rounded-xl border border-white/[0.08] p-5">
              <h3 className="text-sm font-semibold text-gray-300">Explicit Satisfaction</h3>
              <p className="text-xs text-gray-600 mb-3">"Got it" vs "Tag crew" ratio</p>
              <div className="text-3xl font-bold text-green-400 font-code mb-4">
                {metrics.explicitSatisfactionRate.toFixed(1)}%
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between items-center pb-2 border-b border-white/[0.06]">
                  <span className="text-gray-500">"Got it, thanks!"</span>
                  <span className="text-white font-semibold font-code">{metrics.gotItCount}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-500">"Tag the crew"</span>
                  <span className="text-white font-semibold font-code">{metrics.tagCrewCount}</span>
                </div>
              </div>
            </div>

            <div className="bg-white/[0.03] rounded-xl border border-white/[0.08] p-5">
              <h3 className="text-sm font-semibold text-gray-300">Engagement Rate</h3>
              <p className="text-xs text-gray-600 mb-3">Feedback button clicks</p>
              <div className="text-3xl font-bold text-blue-400 font-code mb-4">
                {metrics.engagementRate.toFixed(1)}%
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between items-center pb-2 border-b border-white/[0.06]">
                  <span className="text-gray-500">Button interactions</span>
                  <span className="text-white font-semibold font-code">{metrics.gotItCount + metrics.needHelpCount}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-500">Total queries</span>
                  <span className="text-white font-semibold font-code">{metrics.totalQueries}</span>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </DashboardLayout>
  );
}
