# Bot Usage Guide - @Sage & @Scout

## Two Bots, Two Specialties

**@Sage** = Technical curriculum support (code, debugging, concepts)
**@Scout** = Program support (LMS, FAQ, deadlines, logistics)

### Why Two Bots?
- **Zero guessing**: YOU choose which bot based on your need
- **Specialized knowledge**: Each bot trained on different content
- **No classification errors**: No AI deciding for you

---

## When to Use Which Bot

### Use @Sage for:
- Code debugging & technical errors
- Lecture content questions
- Algorithm explanations
- Project implementation help
- "How do I implement X in Python?"
- "Why is my React component not rendering?"

### Use @Scout for:
- LMS navigation issues
- Submission deadlines
- Program policies & FAQs
- Account/login problems
- "When is Assignment 3 due?"
- "How do I access lecture recordings?"

---

## How to Use (Forum Threads)

1. **Create/open a forum thread** in relevant channel
2. **Tag the right bot** in your message:
   - `@Sage` for tech questions
   - `@Scout` for program questions
3. **Ask your question** clearly
4. **Wait for response** (~5-10 seconds)

### Example Usage

**Technical Question:**
```
@Sage I'm getting "TypeError: list indices must be integers"
in my sorting algorithm. Here's my code: [paste code]
```

**Program Question:**
```
@Scout Where can I find the submission link for Week 4
assignment? I don't see it in LMS.
```

---

## New Features

### 1. Upgraded Knowledge Context
- **Sage**: Full curriculum content (~396KB) - all lectures, code examples, concepts
- **Scout**: Complete FAQ database - program policies, deadlines, common issues
- Both use RAG (Retrieval-Augmented Generation) for accurate answers

### 2. Smart Feedback System
After bot responds, you'll see buttons:

```
🎯 Does this clear things up?
[✅ Got it, thanks!]  [🔄 Need more help]
```

**If satisfied:** Click "Got it, thanks!" → Done
**If need help:** Click "Need more help" → Choose:
- `[💬 Continue here]` = Keep talking to bot
- `[🏴 Tag the crew]` = Escalate to @mekashi/@omkar

### 3. Conversation Memory
- Bots remember last 4 exchanges per thread
- No need to repeat context
- Clarifying questions tracked (max 1 per thread)

### 4. Image Support (@Sage)
Attach screenshots of:
- Error messages
- Code snippets
- Diagrams
- Sage analyzes images via OpenAI Vision

---

## Pro Tips

✅ **Be specific**: "My FastAPI endpoint returns 500" > "API not working"
✅ **Include context**: Error messages, code snippets, what you tried
✅ **Use right bot**: Save time by tagging correct bot first
✅ **Try bot first**: 92% of questions solved without mentor help
✅ **Use feedback buttons**: Helps improve bot responses

❌ **Don't spam both bots**: Pick one based on question type
❌ **Don't DM bots**: Only works in forum threads
❌ **Don't expect instant replies to follow-ups**: Give bot 5-10 sec

---

## Quick Reference

| Question Type | Bot to Tag | Example |
|--------------|-----------|---------|
| Code error | @Sage | "TypeError in line 23" |
| Concept explanation | @Sage | "Explain async/await" |
| Submission deadline | @Scout | "When is A3 due?" |
| LMS issue | @Scout | "Can't access Week 5" |
| Mentor escalation | Either | Click "Tag the crew" button |

---

## Escalation Path

```
Ask bot → Bot responds → Use feedback buttons
                              ↓
                    [Need more help?]
                              ↓
              [Continue here] or [Tag the crew]
                              ↓
              Mentors notified (@mekashi/@omkar)
```

**92% of questions resolved at bot level** - Try bot first!

---

## Common Scenarios

### Scenario 1: Debugging Code
```
@Sage I'm getting this error in my Flask app:
"werkzeug.routing.BuildError: Could not build url for endpoint 'home'"

Here's my route definition: [paste code]
```

### Scenario 2: Understanding Concept
```
@Sage Can you explain how gradient descent works in neural
networks? I don't understand the backpropagation step.
```

### Scenario 3: Program Logistics
```
@Scout Is there a makeup session for Week 3 live class?
I had to miss it due to emergency.
```

### Scenario 4: Multi-Step Help
```
You: @Sage How do I set up Docker for my project?
Sage: [Explains setup steps]
You: [Clicks "Need more help"]
You: [Clicks "Continue here"]
You: I got error "port 5000 already in use"
Sage: [Explains port conflict resolution]
```

---

**Questions about bots?** Tag @Scout with "How do bots work?"
**Found a bug?** Report in #feedback channel or tag mentors.

---

*Last Updated: 2026-02-06*
*Bot Version: Dual-bot v2.0*
