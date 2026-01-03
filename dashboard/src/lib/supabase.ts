import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

export type AnalyticsEvent = {
  id: number
  created_at: string
  event_type: 'query' | 'got_it' | 'tag_crew' | 'need_help' | 'continue_here'
  thread_id: number
  user_id: number
  message_id: number | null
  metadata: Record<string, any> | null
}
