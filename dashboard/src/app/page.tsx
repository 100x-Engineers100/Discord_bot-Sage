"use client";

import { Scene } from "@/components/ui/neon-raymarcher";
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
  const [timeRange, setTimeRange] = useState<7 | 30>(30);

  useEffect(() => {
    fetchAnalytics();
  }, [timeRange]);

  const fetchAnalytics = async () => {
    setLoading(true);
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - timeRange);

    try {
      const { data, error } = await supabase
        .from("analytics_events")
        .select("*")
        .gte("created_at", startDate.toISOString());

      if (error) throw error;

      const events = data as AnalyticsEvent[];

      // Calculate metrics
      const totalQueries = events.filter((e) => e.event_type === "query").length;
      const gotItCount = events.filter((e) => e.event_type === "got_it").length;
      const tagCrewCount = events.filter((e) => e.event_type === "tag_crew").length;
      const needHelpCount = events.filter((e) => e.event_type === "need_help").length;
      const continueCount = events.filter((e) => e.event_type === "continue_here").length;

      // New metric calculations
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

      // Calculate weekly trend
      const trend: TrendData = [];
      const weeks = Math.ceil(timeRange / 7);

      for (let i = 0; i < weeks; i++) {
        const weekStart = new Date(startDate);
        weekStart.setDate(weekStart.getDate() + i * 7);
        const weekEnd = new Date(weekStart);
        weekEnd.setDate(weekEnd.getDate() + 7);

        const weekEvents = events.filter((e) => {
          const eventDate = new Date(e.created_at);
          return eventDate >= weekStart && eventDate < weekEnd;
        });

        trend.push({
          date: `Week ${i + 1}`,
          queries: weekEvents.filter((e) => e.event_type === "query").length,
          gotIt: weekEvents.filter((e) => e.event_type === "got_it").length,
          tagCrew: weekEvents.filter((e) => e.event_type === "tag_crew").length,
        });
      }

      setTrendData(trend);
    } catch (error) {
      console.error("Error fetching analytics:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative w-screen h-screen overflow-hidden">
      {/* 3D Neon Background */}
      <div className="absolute inset-0 z-0">
        <Scene />
      </div>

      {/* Dashboard Content */}
      <div className="relative z-10 w-full h-full overflow-y-auto">
        <div className="container mx-auto px-6 py-8 max-w-7xl">
          {/* Header */}
          <header className="mb-12">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-5xl font-bold tracking-tight text-white mb-2">
                  Sage Analytics
                </h1>
                <p className="text-lg text-green-300/80">
                  100xEngineers AI Cohort 6 · Support Bot Metrics
                </p>
              </div>

              {/* Time Range Selector */}
              <div className="flex gap-2 bg-black/40 backdrop-blur-md rounded-lg p-1 border border-green-500/30">
                <button
                  onClick={() => setTimeRange(7)}
                  className={`px-6 py-2 rounded-md font-medium transition-all ${
                    timeRange === 7
                      ? "bg-green-500/30 text-green-300 shadow-lg shadow-green-500/20"
                      : "text-gray-400 hover:text-white"
                  }`}
                >
                  7 Days
                </button>
                <button
                  onClick={() => setTimeRange(30)}
                  className={`px-6 py-2 rounded-md font-medium transition-all ${
                    timeRange === 30
                      ? "bg-green-500/30 text-green-300 shadow-lg shadow-green-500/20"
                      : "text-gray-400 hover:text-white"
                  }`}
                >
                  30 Days
                </button>
              </div>
            </div>
          </header>

          {loading ? (
            <div className="flex items-center justify-center h-64">
              <div className="text-2xl text-green-300 animate-pulse">Loading analytics...</div>
            </div>
          ) : (
            <>
              {/* Hero Metrics Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
                {/* Self-Service Success */}
                <div className="bg-black/40 backdrop-blur-md rounded-2xl border border-green-500/30 p-6 shadow-2xl hover:shadow-green-500/20 transition-all">
                  <div className="flex items-center gap-4 mb-4">
                    <div className="p-3 bg-green-500/20 rounded-xl">
                      <CheckCircle2 className="w-8 h-8 text-green-400" />
                    </div>
                    <div>
                      <p className="text-sm text-gray-400 font-medium">Self-Service Success</p>
                      <p className="text-xs text-gray-500">Queries resolved without mentor help</p>
                    </div>
                  </div>
                  <div className="text-5xl font-bold text-green-400 mb-2">
                    {metrics.selfServiceRate.toFixed(1)}%
                  </div>
                  <div className="flex items-center gap-2 text-sm text-gray-400">
                    <span>{metrics.totalQueries - metrics.tagCrewCount} of {metrics.totalQueries} queries</span>
                  </div>
                </div>

                {/* Total Queries */}
                <div className="bg-black/40 backdrop-blur-md rounded-2xl border border-green-500/30 p-6 shadow-2xl hover:shadow-green-500/20 transition-all">
                  <div className="flex items-center gap-4 mb-4">
                    <div className="p-3 bg-blue-500/20 rounded-xl">
                      <Activity className="w-8 h-8 text-blue-400" />
                    </div>
                    <div>
                      <p className="text-sm text-gray-400 font-medium">Total Queries</p>
                      <p className="text-xs text-gray-500">Bot responses sent to students</p>
                    </div>
                  </div>
                  <div className="text-5xl font-bold text-white mb-2">{metrics.totalQueries}</div>
                  <div className="flex items-center gap-2 text-sm text-gray-400">
                    <TrendingUp className="w-4 h-4" />
                    <span>Last {timeRange} days</span>
                  </div>
                </div>

                {/* Mentor Escalations */}
                <div className="bg-black/40 backdrop-blur-md rounded-2xl border border-green-500/30 p-6 shadow-2xl hover:shadow-green-500/20 transition-all">
                  <div className="flex items-center gap-4 mb-4">
                    <div className="p-3 bg-amber-500/20 rounded-xl">
                      <AlertTriangle className="w-8 h-8 text-amber-400" />
                    </div>
                    <div>
                      <p className="text-sm text-gray-400 font-medium">Mentor Escalations</p>
                      <p className="text-xs text-gray-500">Students who tagged mentors for help</p>
                    </div>
                  </div>
                  <div className="flex items-baseline gap-3 mb-2">
                    <span className="text-5xl font-bold text-white">{metrics.tagCrewCount}</span>
                    <span className="text-3xl font-semibold text-amber-400">
                      {metrics.escalationRate.toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-gray-400">
                    <span>Of total queries</span>
                  </div>
                </div>
              </div>

              {/* Weekly Trend Chart */}
              <div className="bg-black/40 backdrop-blur-md rounded-2xl border border-green-500/30 p-8 shadow-2xl">
                <h2 className="text-2xl font-bold text-white mb-6">Activity Trend</h2>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={trendData}>
                    <XAxis
                      dataKey="date"
                      stroke="#4ade80"
                      strokeOpacity={0.5}
                      style={{ fontSize: '14px' }}
                    />
                    <YAxis
                      stroke="#4ade80"
                      strokeOpacity={0.5}
                      style={{ fontSize: '14px' }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'rgba(0, 0, 0, 0.9)',
                        border: '1px solid rgba(74, 222, 128, 0.3)',
                        borderRadius: '12px',
                        padding: '12px',
                      }}
                      labelStyle={{ color: '#fff', fontWeight: 'bold', marginBottom: '8px' }}
                      itemStyle={{ color: '#fff' }}
                    />
                    <Legend
                      wrapperStyle={{ paddingTop: '20px' }}
                      iconType="circle"
                    />
                    <Line
                      type="monotone"
                      dataKey="queries"
                      stroke="#4ade80"
                      strokeWidth={3}
                      dot={{ fill: '#4ade80', r: 5 }}
                      name="Total Queries"
                    />
                    <Line
                      type="monotone"
                      dataKey="gotIt"
                      stroke="#22d3ee"
                      strokeWidth={3}
                      dot={{ fill: '#22d3ee', r: 5 }}
                      name="Got It Thanks"
                    />
                    <Line
                      type="monotone"
                      dataKey="tagCrew"
                      stroke="#fbbf24"
                      strokeWidth={3}
                      dot={{ fill: '#fbbf24', r: 5 }}
                      name="Tagged Crew"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Secondary Metrics */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-12">
                <div className="bg-black/40 backdrop-blur-md rounded-2xl border border-green-500/30 p-6">
                  <h3 className="text-lg font-semibold text-white mb-2">Explicit Satisfaction Rate</h3>
                  <p className="text-xs text-gray-500 mb-4">Users who clicked "Got it" vs "Tag crew"</p>
                  <div className="flex items-baseline gap-3 mb-4">
                    <span className="text-4xl font-bold text-green-400">
                      {metrics.explicitSatisfactionRate.toFixed(1)}%
                    </span>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">"Got it, thanks!"</span>
                      <span className="text-white font-bold">{metrics.gotItCount}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">"Tag the crew"</span>
                      <span className="text-white font-bold">{metrics.tagCrewCount}</span>
                    </div>
                  </div>
                </div>

                <div className="bg-black/40 backdrop-blur-md rounded-2xl border border-green-500/30 p-6">
                  <h3 className="text-lg font-semibold text-white mb-2">Engagement Rate</h3>
                  <p className="text-xs text-gray-500 mb-4">Users who clicked any feedback button</p>
                  <div className="flex items-baseline gap-3 mb-4">
                    <span className="text-4xl font-bold text-blue-400">
                      {metrics.engagementRate.toFixed(1)}%
                    </span>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Button interactions</span>
                      <span className="text-white font-bold">{metrics.gotItCount + metrics.needHelpCount}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Total queries</span>
                      <span className="text-white font-bold">{metrics.totalQueries}</span>
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
