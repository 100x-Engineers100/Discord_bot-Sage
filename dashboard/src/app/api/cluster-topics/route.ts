import { NextResponse } from "next/server";
import { unstable_cache } from "next/cache";
import { createClient } from "@supabase/supabase-js";
import OpenAI from "openai";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type QueryEvent = {
  query_text: string;
  thread_id: string;
  created_at: string;
  matched_lectures?: Array<{
    lecture_name: string;
    module_name: string;
    lecture_num: number;
    module: number;
    rag_score: number;
  }>;
};

type TopicAccumulator = {
  query_count: number;
  escalation_count: number;
  firstHalf: number;
  secondHalf: number;
  query_texts: Set<string>;
};

// ---------------------------------------------------------------------------
// Core logic
// ---------------------------------------------------------------------------

const fetchAndCluster = unstable_cache(
  async (days: string) => {
    const supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
    );

    const startDate = new Date();
    startDate.setDate(startDate.getDate() - Number(days));
    const now = new Date();
    const midpoint = new Date(
      startDate.getTime() + (now.getTime() - startDate.getTime()) / 2
    );

    // Paginate to bypass Supabase 1000-row default limit
    const PAGE = 1000;
    let allEvents: any[] = [];
    let from = 0;
    while (true) {
      const { data, error } = await supabase
        .from("analytics_events")
        .select("*")
        .in("event_type", ["query", "tag_crew"])
        .gte("created_at", startDate.toISOString())
        .order("created_at", { ascending: true })
        .range(from, from + PAGE - 1);
      if (error) throw new Error(error.message);
      allEvents = [...allEvents, ...(data ?? [])];
      if (!data || data.length < PAGE) break;
      from += PAGE;
    }

    // -----------------------------------------------------------------------
    // Build escalated threads set + split query events into two buckets
    // -----------------------------------------------------------------------
    const escalatedThreads = new Set<string>();
    const withLectures: QueryEvent[] = [];   // have matched_lectures in metadata
    const withoutLectures: QueryEvent[] = []; // old prod data — need LLM

    for (const e of allEvents) {
      if (e.event_type === "tag_crew") {
        escalatedThreads.add(String(e.thread_id));
        continue;
      }
      if (e.event_type !== "query" || !e.metadata?.query_text) continue;

      const qe: QueryEvent = {
        query_text: e.metadata.query_text as string,
        thread_id: String(e.thread_id),
        created_at: e.created_at as string,
        matched_lectures: e.metadata.matched_lectures,
      };

      if (
        Array.isArray(e.metadata.matched_lectures) &&
        e.metadata.matched_lectures.length > 0
      ) {
        withLectures.push(qe);
      } else {
        withoutLectures.push(qe);
      }
    }

    const topicMap = new Map<string, TopicAccumulator>();

    const upsertTopic = (key: string, qe: QueryEvent) => {
      if (!topicMap.has(key)) {
        topicMap.set(key, {
          query_count: 0,
          escalation_count: 0,
          firstHalf: 0,
          secondHalf: 0,
          query_texts: new Set(),
        });
      }
      const acc = topicMap.get(key)!;
      acc.query_count++;
      acc.query_texts.add(qe.query_text);
      if (escalatedThreads.has(qe.thread_id)) acc.escalation_count++;
      if (new Date(qe.created_at) >= midpoint) acc.secondHalf++;
      else acc.firstHalf++;
    };

    // -----------------------------------------------------------------------
    // Path 1: group by top matched lecture (no LLM needed)
    // -----------------------------------------------------------------------
    for (const qe of withLectures) {
      const top = qe.matched_lectures![0];
      const key = `${top.module_name} — ${top.lecture_name}`;
      upsertTopic(key, qe);
    }

    // -----------------------------------------------------------------------
    // Path 2: LLM clustering for old queries without matched_lectures
    // Cap at 100 unique texts to stay within token budget
    // -----------------------------------------------------------------------
    if (withoutLectures.length > 0) {
      // Deduplicate by query_text
      const uniqueOld = new Map<string, QueryEvent[]>();
      for (const qe of withoutLectures) {
        if (!uniqueOld.has(qe.query_text)) uniqueOld.set(qe.query_text, []);
        uniqueOld.get(qe.query_text)!.push(qe);
      }

      const uniqueTexts = Array.from(uniqueOld.keys()).slice(0, 100);

      try {
        const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });
        const numberedList = uniqueTexts.map((q, i) => `${i}: ${q}`).join("\n");

        const completion = await openai.chat.completions.create({
          model: "gpt-4.1-mini",
          messages: [
            {
              role: "system",
              content: `Group student Discord questions from a coding bootcamp by topic.
Rules:
- Same underlying issue/concept = one group
- Topic name: 3-4 words, specific (e.g. "Virtual Env Setup", "Supabase Connection")
- Every index must appear in exactly one group
- Vague/random questions: one group named "Unclear Questions"
Return ONLY valid JSON, no markdown:
[{"topic":"Topic Name","indices":[0,2,5]},...]`,
            },
            { role: "user", content: numberedList },
          ],
          temperature: 0.2,
          max_tokens: 3000,
        });

        const raw = (completion.choices[0].message.content || "").trim();
        const cleaned = raw.replace(/^```(?:json)?\s*/, "").replace(/\s*```$/, "");
        const clusters: Array<{ topic: string; indices: number[] }> = JSON.parse(cleaned);

        for (const cluster of clusters) {
          for (const idx of cluster.indices) {
            if (idx < 0 || idx >= uniqueTexts.length) continue;
            const qText = uniqueTexts[idx];
            for (const qe of uniqueOld.get(qText) ?? []) {
              upsertTopic(cluster.topic, qe);
            }
          }
        }
      } catch {
        // LLM failed — dump into a catch-all rather than "Uncategorized"
        for (const [, events] of uniqueOld) {
          for (const qe of events) {
            upsertTopic("Other Questions", qe);
          }
        }
      }
    }

    // -----------------------------------------------------------------------
    // Build final topic list
    // -----------------------------------------------------------------------
    const topics = Array.from(topicMap.entries()).map(([topic, acc]) => ({
      topic,
      query_count: acc.query_count,
      escalation_count: acc.escalation_count,
      query_texts: Array.from(acc.query_texts).slice(0, 8), // cap display to 8
      is_trending:
        acc.secondHalf > 0 &&
        (acc.firstHalf === 0 || acc.secondHalf / acc.firstHalf > 1.5),
    }));

    topics.sort((a, b) => b.query_count - a.query_count);
    return { topics };
  },
  ["cluster-topics"],
  { revalidate: 300 }
);

// ---------------------------------------------------------------------------
// Route handler
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
