import { NextResponse } from "next/server";
import { unstable_cache } from "next/cache";
import { createClient } from "@supabase/supabase-js";
import OpenAI from "openai";

// ---------------------------------------------------------------------------
// Core logic — wrapped in unstable_cache so Next.js / Vercel caches it.
// Throws on Supabase error so the error is NOT cached.
// ---------------------------------------------------------------------------
const fetchAndCluster = unstable_cache(
  async (days: string) => {
    const supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
    );

    const startDate = new Date();
    startDate.setDate(startDate.getDate() - Number(days));

    const { data: events, error } = await supabase
      .from("analytics_events")
      .select("*")
      .in("event_type", ["query", "tag_crew"])
      .gte("created_at", startDate.toISOString())
      .order("created_at", { ascending: true });

    if (error) throw new Error(error.message);

    // -----------------------------------------------------------------------
    // Separate escalated threads + extract query texts
    // -----------------------------------------------------------------------
    const escalatedThreads = new Set<string>();
    const queryEvents: Array<{
      query_text: string;
      thread_id: string;
      created_at: string;
    }> = [];

    for (const e of events ?? []) {
      if (e.event_type === "tag_crew") {
        escalatedThreads.add(String(e.thread_id));
      } else if (e.event_type === "query" && e.metadata?.query_text) {
        queryEvents.push({
          query_text: e.metadata.query_text as string,
          thread_id: String(e.thread_id),
          created_at: e.created_at as string,
        });
      }
    }

    if (queryEvents.length === 0) return { topics: [] };

    // -----------------------------------------------------------------------
    // Deduplicate query_texts for the LLM prompt.
    // Track per-unique-query: every occurrence's thread_id + timestamp.
    // -----------------------------------------------------------------------
    const queryMap = new Map<
      string,
      Array<{ thread_id: string; created_at: string }>
    >();

    for (const q of queryEvents) {
      if (!queryMap.has(q.query_text)) queryMap.set(q.query_text, []);
      queryMap.get(q.query_text)!.push({
        thread_id: q.thread_id,
        created_at: q.created_at,
      });
    }

    const uniqueQueries = Array.from(queryMap.keys());

    // -----------------------------------------------------------------------
    // LLM clustering
    // -----------------------------------------------------------------------
    const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });
    const numberedList = uniqueQueries.map((q, i) => `${i}: ${q}`).join("\n");

    const completion = await openai.chat.completions.create({
      model: "gpt-4.1-mini",
      messages: [
        {
          role: "system",
          content: `You are grouping student questions from a coding bootcamp by semantic similarity.

Rules:
- Questions about the same underlying issue or concept go in one group, regardless of wording
- Generate a concise topic name (3-4 words max) that is the CRUX of what students are struggling with
- Be specific: "Virtual Env Setup" beats "Python Basics". "Supabase Connection" beats "Database Issues"
- Every question index MUST appear in exactly one group. No index left out, no duplicates across groups
- If questions are too vague, incomplete, or random to fit any real topic, put ALL of them into a single group with the topic name "Unclear Questions". Do NOT create multiple "Unclear" groups — only one

Return ONLY valid JSON. No markdown fences, no explanation:
[{"topic":"Topic Name","indices":[0,2,5]},...]`,
        },
        {
          role: "user",
          content: numberedList,
        },
      ],
      temperature: 0.2,
      max_tokens: 2000,
    });

    // -----------------------------------------------------------------------
    // Parse + fallback
    // -----------------------------------------------------------------------
    let clusters: Array<{ topic: string; indices: number[] }> = [];
    try {
      const raw = (completion.choices[0].message.content || "").trim();
      const cleaned = raw
        .replace(/^```(?:json)?\s*/, "")
        .replace(/\s*```$/, "");
      clusters = JSON.parse(cleaned);
    } catch {
      clusters = [
        { topic: "Uncategorized", indices: uniqueQueries.map((_, i) => i) },
      ];
    }

    // -----------------------------------------------------------------------
    // Merge any clusters GPT returned with the same topic name.
    // Defensive: prompt says don't do this, but LLMs aren't perfect.
    // -----------------------------------------------------------------------
    const mergedMap = new Map<string, number[]>();
    for (const cluster of clusters) {
      const existing = mergedMap.get(cluster.topic);
      if (existing) {
        existing.push(...cluster.indices);
      } else {
        mergedMap.set(cluster.topic, [...cluster.indices]);
      }
    }
    clusters = Array.from(mergedMap.entries()).map(([topic, indices]) => ({
      topic,
      indices: [...new Set(indices)],
    }));

    // -----------------------------------------------------------------------
    // Build topic entries with counts + trending
    // -----------------------------------------------------------------------
    const now = new Date();
    const midpoint = new Date(
      startDate.getTime() + (now.getTime() - startDate.getTime()) / 2
    );

    const topics = clusters.map((cluster) => {
      let query_count = 0;
      let escalation_count = 0;
      let firstHalf = 0;
      let secondHalf = 0;
      const query_texts: string[] = [];

      for (const idx of cluster.indices) {
        if (idx < 0 || idx >= uniqueQueries.length) continue;

        const qText = uniqueQueries[idx];
        const occurrences = queryMap.get(qText) || [];
        query_texts.push(qText);
        query_count += occurrences.length;

        for (const occ of occurrences) {
          if (escalatedThreads.has(occ.thread_id)) escalation_count++;
          if (new Date(occ.created_at) >= midpoint) secondHalf++;
          else firstHalf++;
        }
      }

      const is_trending =
        secondHalf > 0 && (firstHalf === 0 || secondHalf / firstHalf > 1.5);

      return { topic: cluster.topic, query_count, escalation_count, query_texts, is_trending };
    });

    topics.sort((a, b) => b.query_count - a.query_count);
    return { topics };
  },
  ["cluster-topics"],
  { revalidate: 300 } // 5 min TTL — works on Vercel edge cache
);

// ---------------------------------------------------------------------------
// Route handler — thin wrapper around the cached function
// ---------------------------------------------------------------------------
export async function GET(request: Request) {
  const url = new URL(request.url);
  const days = url.searchParams.get("days") || "30";

  try {
    const result = await fetchAndCluster(days);
    return NextResponse.json(result);
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
