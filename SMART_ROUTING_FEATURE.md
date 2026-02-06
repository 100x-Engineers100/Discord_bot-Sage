# Smart Bot Routing Feature

## What Was Added

Implemented **smart bot routing** that detects when users tag the wrong bot and redirects them to the correct one.

---

## How It Works

### Two-Layer Detection

**Layer 1: Fast Keyword Check** (runs before RAG retrieval)
- Checks query against FAQ keywords (lms, launchpad, jarvis, recordings, etc.)
- Checks query against technical keywords (controlnet, rag, agent, error, code, etc.)
- If clear mismatch → instant redirect (no API call wasted)

**Layer 2: RAG Score Fallback** (runs after RAG retrieval)
- Checks RAG similarity score (FAISS L2 distance)
- If score > 1.5 (poor match) → redirect to other bot
- If no results found → redirect to other bot

---

## Examples

### Example 1: Sage Asked FAQ Question

```
Student: @Sage where can I find session recordings?

Sage: Hey! This looks like a program logistics question.
      I'm Sage - I specialize in technical curriculum help
      (code, concepts, debugging).

      For questions about LMS, recordings, Launchpad, Jarvis Labs,
      or session timings, try asking @Scout - they handle all
      program support! 🧭
```

**Detection**: Keyword match ("recordings", "session" = FAQ keywords)

---

### Example 2: Scout Asked Technical Question

```
Student: @Scout how does ControlNet work in ComfyUI?

Scout: Hey! This looks like a technical question.
       I'm Scout - I handle program logistics (LMS, recordings,
       Jarvis Labs, Launchpad).

       For technical help with curriculum concepts, code, or
       debugging, try asking @Sage - they're the tech expert! 🤖
```

**Detection**: Keyword match ("ControlNet", "ComfyUI" = technical keywords)

---

### Example 3: Poor RAG Match Fallback

```
Student: @Sage what's the Jarvis Labs team code?

Sage: [RAG searches curriculum, finds no good match - score 2.8]

Sage: Hmm, I couldn't find relevant info in the curriculum docs.
      If you're asking about program logistics (LMS, recordings,
      Jarvis Labs), try @Scout instead! 🧭

      I'm better at technical questions about curriculum
      concepts and code.
```

**Detection**: Poor RAG score (2.8 > 1.5 threshold)

---

## When Routing DOESN'T Trigger

**Ambiguous queries** (uncertain classification):
```
Student: @Sage what is Module 2 about?
Sage: [Normal response about Module 2: LLM Full Stack]
```

No strong keywords detected → let Sage try to answer.

**Good matches** (appropriate bot):
```
Student: @Sage how does RAG work?
Sage: [Normal technical response with curriculum context]
```

Technical keywords + good RAG score → answer normally.

---

## Technical Implementation

### Files Modified

1. **`bot.py`**
   - Updated `retrieve_curriculum_context()` to return tuple: `(context, rag_results)`
   - Updated `retrieve_faq_context()` to return tuple: `(context, rag_results)`
   - Added routing check in Sage's `on_message` handler
   - Added routing check in Scout's `on_message` handler

2. **`bot_routing_utils.py`** (NEW)
   - `detect_query_type()`: Keyword-based classification
   - `should_suggest_other_bot()`: Combined detection logic
   - `generate_redirect_message()`: User-friendly redirect messages

### Architecture

```
User tags @Sage
    ↓
Extract query
    ↓
Retrieve curriculum context (get RAG results with scores)
    ↓
Check: should_suggest_other_bot('sage', query, rag_results)
    ↓
    ├─→ Redirect to Scout (wrong bot detected)
    └─→ Generate normal Sage response
```

---

## Test Results

```bash
$ python test_bot_routing.py
```

**Output**:
```
[PASS] Keyword Detection (13/13 tests)
[PASS] Wrong-Bot Detection (6/6 tests)
[PASS] Redirect Messages (3/3 tests)

[OK] All routing tests passed!
```

**Test Coverage**:
- FAQ queries to Sage → Redirects to Scout ✓
- Technical queries to Scout → Redirects to Sage ✓
- Poor RAG scores → Redirects to other bot ✓
- Good matches → No redirect ✓
- Ambiguous queries → No redirect (let bot try) ✓

---

## Keyword Lists

### FAQ Keywords (43 keywords)
- **LMS**: lms, login, access, dashboard, password, account
- **Sessions**: recording, session, lecture timing, calendar, live session
- **Launchpad**: launchpad, form, track, discovery call, ikigai
- **Jarvis**: jarvis, credits, instance, gpu, team code
- **Logistics**: mail id, email, contact, announcements, support
- **Prerequisites**: prerequisite, python study, beginner

### Technical Keywords (60+ keywords)
- **Module 1**: diffusion, controlnet, ip adapter, comfyui, flux, lora, sdxl
- **Module 2**: llm, rag, retrieval, vector, embedding, faiss, langchain, fastapi
- **Module 3**: agent, multi-agent, agentic, workflow, mcp, guardrails
- **Code**: error, bug, debug, code, implement, api, function, import
- **Concepts**: architecture, model, training, inference, transformer

---

## Thresholds & Tuning

### RAG Score Threshold
- **Good match**: score < 1.0
- **Okay match**: 1.0 - 1.5
- **Poor match**: > 1.5 (triggers redirect)

Based on testing with real queries. Adjust in `bot_routing_utils.py` if needed.

### Keyword Match Threshold
- **Strong signal**: 2+ keyword matches
- **Single strong keyword**: Unambiguous terms (e.g., "launchpad", "controlnet")
- **Uncertain**: < 2 matches → let bot try

---

## User Experience Flow

### Happy Path (Correct Bot)

```
Student: @Sage how does ControlNet work?
→ Sage responds normally with curriculum context
→ Feedback buttons appear
```

### Redirect Path (Wrong Bot)

```
Student: @Sage where are session recordings?
→ Sage detects FAQ query
→ Immediate redirect message (no wasted API call)
→ Student asks @Scout instead
→ Scout provides FAQ answer
```

### Uncertain Path

```
Student: @Sage what is Module 2 about?
→ No strong keywords detected
→ Sage tries to answer from curriculum
→ If good match: answers normally
→ If poor match: suggests Scout as fallback
```

---

## Performance Impact

### Minimal Overhead

- **Fast keyword check**: ~0.001s (negligible)
- **RAG score check**: Already computed during retrieval (no extra cost)
- **No extra API calls**: Detection uses existing RAG results

### Cost: $0

- No additional OpenAI API calls
- Uses keywords + existing RAG scores
- Pure logic-based detection

---

## Deployment

### Already Integrated

Smart routing is **already integrated** into bot.py. No additional deployment steps needed!

### Testing in Production

After deploying bot.py:

**Test 1: Sage → Scout redirect**
```
@Sage where can I find session recordings?
```
Expected: Redirect message suggesting @Scout

**Test 2: Scout → Sage redirect**
```
@Scout how does ControlNet work in ComfyUI?
```
Expected: Redirect message suggesting @Sage

**Test 3: Normal Sage query**
```
@Sage explain RAG architecture
```
Expected: Normal technical answer (no redirect)

**Test 4: Normal Scout query**
```
@Scout how do I access Jarvis Labs?
```
Expected: Normal FAQ answer (no redirect)

---

## Maintenance

### Adding New Keywords

To add FAQ keywords (e.g., new program feature):

**File**: `bot_routing_utils.py`

```python
FAQ_KEYWORDS = [
    # ... existing keywords ...
    'new_feature', 'new_platform', 'new_tool'  # Add here
]
```

To add technical keywords (e.g., new curriculum module):

```python
TECHNICAL_KEYWORDS = [
    # ... existing keywords ...
    'new_model', 'new_framework', 'new_concept'  # Add here
]
```

### Adjusting RAG Threshold

If getting too many false redirects:

**File**: `bot_routing_utils.py`, line ~155

```python
if best_score > 1.5:  # Increase this (e.g., 2.0) to be more lenient
    return (True, 'poor_match')
```

If missing obvious wrong-bot queries:

```python
if best_score > 1.0:  # Decrease this to be stricter
    return (True, 'poor_match')
```

---

## Summary

### What You Get

✅ **Smart detection**: Catches wrong-bot queries before wasting API calls
✅ **User-friendly**: Clear redirect messages explaining which bot to use
✅ **Two-layer**: Fast keywords + RAG score fallback
✅ **Zero cost**: No extra API calls
✅ **Tested**: 22/22 tests passing
✅ **Production-ready**: Already integrated into bot.py

### User Impact

**Before**:
```
@Sage where are recordings?
Sage: [tries to answer, fails, gives generic "I don't know" response]
Student: frustrated, tags mentor
```

**After**:
```
@Sage where are recordings?
Sage: This looks like a program question! Try @Scout - they handle LMS/recordings.
Student: @Scout where are recordings?
Scout: [step-by-step instructions]
Student: happy!
```

**Result**: Better UX, fewer mentor tags, clearer bot separation.

---

## Next Steps

1. **Deploy bot.py** (routing already integrated)
2. **Test in Discord** (use test cases above)
3. **Monitor for false positives** (adjust thresholds if needed)
4. **Update keywords** as curriculum/program evolves

No additional work needed - feature is ready to go! 🚀
