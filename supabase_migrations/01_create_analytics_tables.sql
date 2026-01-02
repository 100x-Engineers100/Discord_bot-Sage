-- ============================================================================
-- Sage Bot Analytics Database Schema
-- ============================================================================
-- This migration creates the core tables for tracking bot analytics
-- Run this in Supabase SQL Editor

-- Table 1: Analytics Events
-- Tracks every interaction with the bot
CREATE TABLE IF NOT EXISTS analytics_events (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    thread_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    message_id BIGINT,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Add comments for clarity
COMMENT ON TABLE analytics_events IS 'Tracks all bot interactions and button clicks';
COMMENT ON COLUMN analytics_events.event_type IS 'Event types: query, got_it, tag_crew, need_help, continue_here';
COMMENT ON COLUMN analytics_events.thread_id IS 'Discord thread ID where event occurred';
COMMENT ON COLUMN analytics_events.user_id IS 'Discord user ID who triggered event';
COMMENT ON COLUMN analytics_events.message_id IS 'Discord message ID (optional)';
COMMENT ON COLUMN analytics_events.metadata IS 'Additional context (JSON)';

-- Create indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_analytics_event_type ON analytics_events(event_type);
CREATE INDEX IF NOT EXISTS idx_analytics_created_at ON analytics_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analytics_thread_id ON analytics_events(thread_id);
CREATE INDEX IF NOT EXISTS idx_analytics_user_id ON analytics_events(user_id);

-- Composite index for dashboard queries (event_type + date range)
CREATE INDEX IF NOT EXISTS idx_analytics_type_date ON analytics_events(event_type, created_at DESC);

-- ============================================================================
-- Success message
-- ============================================================================
DO $$
BEGIN
    RAISE NOTICE '✅ Analytics tables created successfully!';
    RAISE NOTICE '📊 Table: analytics_events';
    RAISE NOTICE '🔍 Indexes created for fast queries';
END $$;
