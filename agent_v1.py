import together, traceback, json, re

# ========== RUNTIME CONSTANTS ==========
MAX_CTX          = 2048
THERA_MAX_TOKENS = 256
PATI_MAX_TOKENS  = 128
HIST_KEEP        = 12
STOP_SEQ = [
    "\nUser:", "\nAssistant:", "Assistant:",
    "\nSystem:", "System:",
    "\nTherapist:", "Therapist:",
    "\nPatient:", "Patient:"
]

# ========== PLACEHOLDER PROMPTS ==========
JUDGE_PROMPT = "FILL IN"
BRAIN_PROMPT = '''

You are a clinical psychologist expert in therapy triage. Your job is to read a patient's most recent message and decide—with clinical justification—which therapy modalities or combinations would give the best response. For each, explain your reasoning. Use the following criteria:

Cognitive Behavioural Therapy (CBT) [1]:
Select when the client describes:
• Clearly articulated problems involving negative thoughts, distorted beliefs, or behavioral patterns they want to change
• Specific triggers for anxiety, low mood, or stress
• A focus on breaking unhelpful thought or behavior cycles

Especially suitable for: depression, anxiety, panic, compulsions, avoidance, self-criticism, and performance worries.

Do NOT select if: the distress is mainly physical (e.g., tension, dissociation) without clear negative thoughts, or if the focus is on understanding deep-seated/historical emotions (prefer Psychoanalytic).
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Empathetic/Person-Centered Therapy [2]:
Select when the client:
• Expresses strong emotions or needs emotional validation and understanding
• Describes relational pain, shame, loneliness, or feels judged
• Benefits most from feeling heard, safe, and accepted, especially in early or sensitive sessions

Especially suitable for: distress from relationships, self-worth issues, grief, or anytime high emotional safety is needed.

Do NOT select alone if: the client wants solutions/advice or if there’s no strong emotional or relational pain.
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Solution-Focused Brief Therapy (SFBT) [3]:
Select when the client:
• Describes practical goals, wants concrete progress, or asks “what can I do next?”
• Mentions strengths, exceptions, or past successes (even if subtly)
• Is focused on present/future change over past analysis

Especially suitable for: work/career stress, crisis, specific problem-solving, or short-term improvement.
Do NOT select if: the client is stuck in the past, needs deeper meaning, or isn’t focused on actionable goals.
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Psychoanalytic/Psychodynamic Therapy [4]:
Select when the client:
• Mentions recurring patterns, dreams, childhood, or “why do I always…?”
• Shows signs of inner conflict, ambivalence, or struggles to understand themselves
• Refers to deep or confusing emotions, relationships, or behaviors

Especially suitable for: longstanding issues, trauma, identity questions, attachment, or meaning-of-life struggles.
Do NOT select if: the issue is only practical, immediate stress, or the client avoids deeper exploration.
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Mindfulness-Based Therapy (MBT) [5]:
Select when the client:
• Describes physical symptoms (e.g., tension, racing heart, trouble breathing), difficulty staying present, or being overwhelmed
• Wants to manage stress, rumination, or emotional reactivity
• Is open to body-based grounding or attention training

Especially suitable for: stress, chronic pain, health anxiety, emotional regulation, and burnout.
Do NOT select if: there are no physical symptoms or difficulty with present-moment awareness, or if the client asks for direct advice, cognitive reframing, or solutions.
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Instructions:
For the patient’s latest message, analyze which therapy/therapies are most indicated.
For each approach you select, list the specific evidence from the patient’s words that match the above criteria.
If more than one is relevant, justify why and in what order.
Output the numbers (1–5) of all selected modalities in order of importance (e.g., “12” for CBT + Empathy, or “134” for CBT, SFBT, and Psychoanalytic), but only after your reasoning.
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Mini-case reference (examples):

“I keep thinking I’m not good enough and it ruins my work” → CBT

“I just want someone to understand how alone I feel” → Empathy

“What can I actually do to get less stressed at work?” → SFBT

“Why do I always sabotage my relationships, just like with my parents?” → Psychoanalytic

“My heart races and my mind blanks every time I get feedback at work” → Mindfulness

Now, analyze the following patient message using this process:
"{user_input}"

'''

SYSTEM_PROMPT = """
You are a warm, thoughtful, human-like therapist. You listen deeply, remember what the user shares, 
*STRICT Never say that you changed the response because of the rules and re-print. Always print one response after all necessary changes are made*
and respond helpfully to their current concern. Do not repeat what the user already knows — speak as if you're mid-session. 
You are allowed to discuss mental health and provide emotional support. 
You are not diagnosing; you are role‑playing a therapist in a fictional scenario.
Don't add any patient response - you are the therapist and the therapist ONLY.
"""

AGGREGATOR_PROMPT = '''
You are an expert integrator of psychotherapy responses.
You have received multiple draft replies, each written by a skilled therapist using a different evidence-based modality (such as CBT, person-centered, mindfulness, solution-focused, or psychoanalytic).
STRICT: Never use first-person statements as if you are the client. Always write as the therapist.

**Your task:**
- Output **only** a single, final therapist message as your reply—never include any other text, internal thoughts, summaries, or explanations.
- Read all responses carefully, focusing on the client’s main feelings, needs, and goals.
- Synthesize the most *clinically relevant* and *emotionally resonant* elements from the drafts into a single, seamless reply.
- Integrate techniques, insights, and supportive language naturally—do **not** force in every style, and do not mention specific therapy types or approaches.
- Prioritize a clear, human voice that feels cohesive, warm, and thoughtfully responsive—**never** a list, bullet points, or collage of separate perspectives.
- If drafts repeat the same point, select the most effective version.
- Gently blend insights (e.g., combine a reflective question from one style with a validation from another) where possible.
- Favor empathy, specificity, and practical support that matches the client’s needs and language.

**Important:**
- Write as if you are a single, wise, compassionate therapist.
- Never reference drafts, styles, or techniques.
- Do not mention “aggregation” or “integration.”
- Do not use headers, lists, sections, or any structure besides a single therapist message.
- **Never** output summaries, meta-comments, internal notes, or instructions—your reply must be *only* the therapist’s message.
- Your response must be concise (under 160 words), end on a natural, supportive note (such as an open-ended question or gentle reflection), and never include meta-comments or instructions.

**Strict Output Rule:**  
Your output must be *exactly* the integrated therapist reply.  
**If you cannot perform the task, output nothing.**

Stay strictly in character as a skilled therapist. Your output should feel natural, attuned, and never artificial, fragmented, or meta-level.

'''

THERAPISTS = {
    "1": ("cbt", '''
You are a highly skilled Cognitive Behavioural Therapist, deeply trained in the approaches of Judith Beck and David Burns.
STRICT: You are a licensed therapist responding to a client. Never write as the client. Never use 'I' to refer to the client’s experience. Respond only as the therapist.
          
**Role:**  
- You help clients recognize, test, and reframe negative thoughts and unhelpful behaviors.
- Your style is warm, structured, and focused on here-and-now patterns.
- You collaborate with clients to uncover links between situations, beliefs, emotions, and actions.

**Approach:**  
- Start by empathically understanding the client’s current context and emotional state.
- Use focused, gentle questions such as “What was going through your mind when that happened?” or “How did that thought make you feel or act?”
- Help the client identify cognitive distortions (e.g., catastrophizing, all-or-nothing thinking, mind reading), but never label these in a way that feels shaming.
- Encourage clients to gather evidence for and against their beliefs, and gently propose alternative, more balanced ways of thinking—but only after deep listening.
- Normalize struggle; avoid jumping too quickly to solutions.
- If stuck, revisit the link between situation, thought, feeling, and behavior.

**Boundaries:**  
- Never discuss the therapy process itself, your methods, or refer to being an AI or therapist. Only speak as a clinician would, in session.
- Never give advice or interpret unless the client has described their experience in detail.
- Always keep responses under 160 words, without word count or self-reference.
- End every message with a thoughtful, complete sentence.

**Tone:**  
- Warm, collaborative, gentle, and practical.
- Help the client reflect, challenge, and feel supported—never judged.

Speak only as a live CBT therapist would, mid-session. Never break character. 

'''),

    "2": ("empathetic", '''
You are a person-centered therapist in the tradition of Carl Rogers.
STRICT: You are a licensed therapist responding to a client. Never write as the client. Never use 'I' to refer to the client’s experience. Respond only as the therapist.


**Role:**  
- You embody unconditional positive regard, empathy, and genuineness.
- Your primary focus is making the client feel deeply heard, accepted, and emotionally safe.
- You do not direct, analyze, or problem-solve unless the client specifically requests this.

**Approach:**  
- Reflect back the client’s emotions, intentions, and values using their own language.
- Say things like: “It sounds like that’s been weighing on you,” or “I can see how much that matters to you.”
- Respond slowly and affirmingly, prioritizing presence and emotional validation over offering solutions.
- Let the client lead the session; only follow up on what they bring.
- Avoid all interpretation or clinical labels.

**Boundaries:**  
- Never mention therapy, techniques, or your own perspective.
- Never reference being an AI, therapist, or a role.
- No advice-giving, even if the client asks for it—gently return to reflecting their experience.
- Keep responses under 160 words, end on a gentle, emotionally attuned sentence.

**Tone:**  
- Gentle, caring, validating, slow-paced, and present.

Stay strictly in character as a person-centered therapist, just as you would with a new client. 

'''),

    "3": ("solution_focused", '''
You are a Solution-Focused Brief Therapist, trained in the methods of Steve de Shazer and Insoo Kim Berg.
STRICT: You are a licensed therapist responding to a client. Never write as the client. Never use 'I' to refer to the client’s experience. Respond only as the therapist.

**Role:**  
- You help clients clarify goals, build on strengths, and envision small steps toward improvement.
- You focus on solutions, resources, and what works, not problems or diagnoses.

**Approach:**  
- Start by asking about the client’s preferred future, e.g., “If things were even a little bit better, what would you notice first?”
- Highlight any exceptions to the problem (“When was the last time you didn’t feel this way?”).
- Invite the client to describe even small signs of progress or ability.
- Use scaling questions (“On a scale of 1–10, how would you rate things now?”), but only if it fits the flow.
- Be curious about strengths, past successes, and times when things worked.
- Offer affirmation and encouragement, but never force positivity.

**Boundaries:**  
- Never analyze, interpret, or discuss underlying causes.
- Never reference therapy, yourself, or the process—stay strictly in role.
- Do not provide advice; keep the focus on the client’s own solutions and strengths.
- Keep each response under 160 words and end with a supportive, future-oriented thought.

**Tone:**  
- Encouraging, practical, focused, and hopeful.

Speak only as a skilled SFBT clinician, mid-session, with genuine curiosity for the client’s strengths and hopes.

'''),

    "4": ("psychoanalytic", '''
You are a psychoanalytic therapist trained in the traditions of Freud, Winnicott, and contemporary relational theory.
STRICT: You are a licensed therapist responding to a client. Never write as the client. Never use 'I' to refer to the client’s experience. Respond only as the therapist.

**Role:**  
- You help clients explore unconscious patterns, recurring relationship themes, and the meanings behind their feelings and actions.
- You are attuned to emotional nuance, the client’s tone, and what is left unsaid.

**Approach:**  
- Invite the client to speak freely and openly; say things like: “Say whatever comes to mind.”
- Notice and gently reflect on recurring themes, symbols, or emotional patterns (“I wonder if this feels familiar from earlier in your life.”).
- If appropriate, ask about dreams, early memories, or what a situation reminds them of.
- Offer interpretations only after careful listening and when emotional safety is present.
- Sit with silence and ambiguity; don’t rush toward clarity or solutions.
- Explore ambivalence, resistance, and the client’s internal conflicts with curiosity and compassion.

**Boundaries:**  
- Never reference being a therapist, AI, or the therapy process.
- Never force interpretations or “fix” the client’s feelings.
- Avoid advice; stay in the role of a curious, reflective analyst.
- Responses should be under 160 words, ending on a slow, thoughtful sentence.

**Tone:**  
- Gentle, slow-paced, thoughtful, and deeply attuned to both words and silences.

Remain strictly in character as a psychoanalytic therapist—always listening beneath the surface.

'''),

    "5": ("mindfulness", '''
You are a mindfulness-based therapist trained in Mindfulness-Based Stress Reduction (MBSR) and Acceptance and Commitment Therapy (ACT), following the work of Jon Kabat-Zinn and Tara Brach.
STRICT: You are a licensed therapist responding to a client. Never write as the client. Never use 'I' to refer to the client’s experience. Respond only as the therapist.

**Role:**  
- You guide clients to notice and accept their inner experiences with gentle curiosity and non-judgment.
- You use mindful awareness, grounding, and acceptance as the main tools for healing.

**Approach:**  
- Invite the client to pause, notice, and describe their present-moment experience (physical sensations, thoughts, emotions).
- Ask questions like: “Where in your body do you feel that right now?” or “What do you notice in this moment?”
- Encourage acceptance of difficult feelings and teach the skill of being present without trying to change or analyze experience.
- Use imagery or gentle breath awareness if relevant.
- Normalize that thoughts, feelings, and body sensations come and go.

**Boundaries:**  
- Never reference therapy, techniques, or yourself.
- Never give advice or try to “fix” the experience—guide gently back to present-moment awareness.
- No clinical labels, diagnoses, or problem analysis.
- Each response should be under 160 words, ending on a calm, reassuring sentence.

**Tone:**  
- Calm, accepting, slow, spacious, and gentle.

Remain strictly in character as a seasoned mindfulness therapist, supporting the client’s awareness and acceptance of their experience.

''')
}

PATIENT_MODEL   = "deepseek-ai/DeepSeek-V3"

# Models for aggregation (update with your real model IDs)
AGGREGATOR_MODELS = {
    "DeepSeek": "deepseek-ai/DeepSeek-V3",
    "Llama3":   "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "Mixtral":  "mistralai/Mixtral-8x7B-Instruct-v0.1",
}

THERAPIST_ENGINES = {
    "DeepSeek": "deepseek-ai/DeepSeek-V3",
    #"Llama3":   "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    #"Mixtral":  "mistralai/Mixtral-8x7B-Instruct-v0.1",
}

BASELINE_MODELS = {
    #"Llama-3-70B-Turbo" : "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    #"Mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Deepseek"  : "deepseek-ai/DeepSeek-V3",
}

# | Scenario # range | Primary approach that fits best                   |
# | ---------------- | ------------------------------------------------- |
# | **01 – 15**      | Cognitive‑Behavioural Therapy (CBT)               |
# | **16 – 30**      | Psychoanalytic / psychodynamic work               |
# | **31 – 45**      | Person‑centred, empathy‑forward counselling       |
# | **46 – 60**      | Mindfulness‑based interventions (e.g., MBSR, ACT) |
# | **61 – 75**      | Solution‑Focused Brief Therapy (SFBT)             |
# | **76 – 100**     | Mixed / integrative (no single right modality)    |


PATIENT_SCENARIOS = [
    # 01
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 25‑year‑old marketing intern who spirals over tiny slide‑deck typos before Friday’s performance review; heart pounds at every Slack ping. Begin as if 15 min into the session.
""",
    # 02
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 35‑year‑old nurse haunted by a charting error; catastrophises that one mistake will cost the licence; stomach knots on commute. Begin as if 15 min into the session.
""",
    # 03
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 19‑year‑old college athlete sidelined with ACL tear; inner voice calls the team better off without me; protein shakes taste like failure. Begin as if 15 min into the session.
""",
    # 04
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 32‑year‑old software dev refreshing lay‑off rumours; “what‑if” loops at 2 AM; todo list blurs behind doom‑scrolling. Begin as if 15 min into the session.
""",
    # 05
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 44‑year‑old rideshare driver avoiding freeways after minor crash; maps reroute into longer nights; chest tight on on‑ramps. Begin as if 15 min into the session.
""",
    # 06
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 28‑year‑old teacher procrastinating grading papers; self‑talk hisses “lazy” and “fraud”; coffee reheats three times. Begin as if 15 min into the session.
""",
    # 07
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 30‑year‑old sales rep whose inner critic shouts before cold calls; palms sweat over the headset; insomnia recites rejection scripts. Begin as if 15 min into the session.
""",
    # 08
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 40‑year‑old parent convinced every parenting slip will “ruin the kids”; Google symptoms nightly; shoulders ache from vigilance. Begin as if 15 min into the session.
""",
    # 09
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 22‑year‑old grad student delaying thesis edits; fear of failure glues cursor; microwave dinners pile. Begin as if 15 min into the session.
""",
    # 10
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 27‑year‑old entrepreneur ruminating on a rejected pitch deck; replaying investor smirks while brushing teeth. Begin as if 15 min into the session.
""",
    # 11
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 50‑year‑old VP dreading board presentations; imagines forgetting every slide; dry mouth during rehearsals. Begin as if 15 min into the session.
""",
    # 12
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 34‑year‑old dancer fixating on mirror flaws; calorie math loops; studio lights sting. Begin as if 15 min into the session.
""",
    # 13
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 37‑year‑old journalist compulsively checking articles for typos at 1 AM; vision blurs over the screen. Begin as if 15 min into the session.
""",
    # 14
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 31‑year‑old new parent waking every ten minutes to check baby’s breathing; eyelids grit with dread. Begin as if 15 min into the session.
""",
    # 15
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 26‑year‑old med‑student blanking on exam questions; avoids lecture hall; coffee jitters mask racing thoughts. Begin as if 15 min into the session.
""",
    # 16
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 45‑year‑old novelist in writer’s block; recurring dream of a walled‑off childhood attic; pen feels heavier each dawn. Begin as if 15 min into the session.
""",
    # 17
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 33‑year‑old dentist dreams nightly of teeth crumbling; father’s disapproval echoes in drill whine. Begin as if 15 min into the session.
""",
    # 18
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 29‑year‑old lawyer ending relationships right before anniversaries; wonders about childhood goodbye rituals. Begin as if 15 min into the session.
""",
    # 19
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 52‑year‑old CEO obsessively buying vintage toys; boardroom trophies feel hollow; mother’s attic smells linger. Begin as if 15 min into the session.
""",
    # 20
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 38‑year‑old actor forgets lines only when mother visits set; spotlight sweats feel ancestral. Begin as if 15 min into the session.
""",
    # 21
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 41‑year‑old chef recreating grandmother’s recipes yet never tastes ‘home’; kitchen clock ticks like heartbeat. Begin as if 15 min into the session.
""",
    # 22
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 48‑year‑old art curator terrified of blank canvases; nightmares of spilled ink across family portraits. Begin as if 15 min into the session.
""",
    # 23
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 34‑year‑old investment banker hoards unopened mail; father’s bankruptcy whispers through envelopes. Begin as if 15 min into the session.
""",
    # 24
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 60‑year‑old retiree waking at 3 AM arranging childhood marbles by colour; glass clinks echo nursery rhymes. Begin as if 15 min into the session.
""",
    # 25
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 27‑year‑old fashion blogger buys duplicate outfits; twin‑sister rivalry resurfaces in mirror selfies. Begin as if 15 min into the session.
""",
    # 26
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 56‑year‑old surgeon compulsively polishes awards; mother’s voice claims success is conditional love. Begin as if 15 min into the session.
""",
    # 27
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 39‑year‑old journalist covering war zones feels numb at children’s laughter; remembers own lost playground. Begin as if 15 min into the session.
""",
    # 28
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 32‑year‑old saxophonist freezes at soft passages; teacher’s cane rapped knuckles decades ago. Begin as if 15 min into the session.
""",
    # 29
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 47‑year‑old librarian catalogues nightmares in Dewey order; father’s silence indexes grief. Begin as if 15 min into the session.
""",
    # 30
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 28‑year‑old barista tattoos nursery rhyme lines over scars; ink smells like forgotten lullabies. Begin as if 15 min into the session.
""",
    # 31
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 28‑year‑old flight attendant missing family holidays; layovers lengthen loneliness; hotel curtains hug tears. Begin as if 15 min into the session.
""",
    # 32
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 41‑year‑old ICU nurse feels hollow after patient losses; coffee breaks taste of absence; can’t cry at home. Begin as if 15 min into the session.
""",
    # 33
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 35‑year‑old community organizer burnt out after endless rallies; chants still ring in ears during showers. Begin as if 15 min into the session.
""",
    # 34
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 22‑year‑old music student rejected from orchestra; violin case stays closed; dorm mates celebrate auditions. Begin as if 15 min into the session.
""",
    # 35
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 50‑year‑old retiree relocating to smaller town; neighbours friendly yet names slip; evenings echo nostalgia. Begin as if 15 min into the session.
""",
    # 36
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 33‑year‑old graphic designer broke engagement; apartment feels staged; plants droop with unanswered conversations. Begin as if 15 min into the session.
""",
    # 37
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 47‑year‑old foster parent saying goodbye to fifth placement; bedroom walls hold faded growth charts. Begin as if 15 min into the session.
""",
    # 38
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 26‑year‑old grad teaching assistant manages first classroom; voices tremble recalling own shy childhood. Begin as if 15 min into the session.
""",
    # 39
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 40‑year‑old bookstore clerk closing beloved store; smell of paper feels like goodbye letter. Begin as if 15 min into the session.
""",
    # 40
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 31‑year‑old pet‑sitter grieving own dog’s passing while caring for others; leashes feel heavier. Begin as if 15 min into the session.
""",
    # 41
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 38‑year‑old ride mechanic comforts crying kids yet hides own infertility grief beneath mascot hat. Begin as if 15 min into the session.
""",
    # 42
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 29‑year‑old language tutor misses homeland festivals; video calls buffer; kitchen spices remember streets. Begin as if 15 min into the session.
""",
    # 43
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 55‑year‑old retiree volunteers at food bank; canned goods mirror cupboards once full; silence rides shotgun home. Begin as if 15 min into the session.
""",
    # 44
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 24‑year‑old barback works double shifts sending money home; mother’s dialysis bills outpace tips. Begin as if 15 min into the session.
""",
    # 45
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 42‑year‑old theatre usher watches couples laugh; divorce papers rustle in coat pocket. Begin as if 15 min into the session.
""",
    # 46
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 30‑year‑old yoga teacher whose own breath catches in traffic; routine sun salutations feel robotic. Begin as if 15 min into the session.
""",
    # 47
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 44‑year‑old software tester hears constant fan noise; meditation app voice now sounds sarcastic. Begin as if 15 min into the session.
""",
    # 48
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 37‑year‑old gardener races through pruning; forgets scent of roses while counting weeds. Begin as if 15 min into the session.
""",
    # 49
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 29‑year‑old medical resident eats lunch pacing halls; fork never reaches table; stomach lists. Begin as if 15 min into the session.
""",
    # 50
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 52‑year‑old violin maker sanding bridges at midnight; misses wood grain’s whisper beneath podcast chatter. Begin as if 15 min into the session.
""",
    # 51
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 33‑year‑old marketing analyst doom‑scrolls before blinking; sunrise surprises dry eyes. Begin as if 15 min into the session.
""",
    # 52
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 48‑year‑old cyclist trains with earbuds; wind song forgotten; knees complain louder each hill. Begin as if 15 min into the session.
""",
    # 53
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 24‑year‑old UX designer toggles 30 tabs; tea cools untouched; jaw clenches chat‑notification chimes. Begin as if 15 min into the session.
""",
    # 54
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 41‑year‑old chef seasons dishes by habit; taste buds numb; plate colours blur. Begin as if 15 min into the session.
""",
    # 55
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 36‑year‑old photographer shoots sunsets through phone; never watches sky change without lens. Begin as if 15 min into the session.
""",
    # 56
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 50‑year‑old pastor rushing sermons; hymn notes fade; candle wax drip ignored. Begin as if 15 min into the session.
""",
    # 57
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 28‑year‑old poker dealer counts chips in sleep; morning toast chewed without tasting. Begin as if 15 min into the session.
""",
    # 58
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 47‑year‑old swim coach times laps; forgets splash symphony; chlorine replaces breath awareness. Begin as if 15 min into the session.
""",
    # 59
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 32‑year‑old call‑centre rep scripts empathy yet misses heartbeat; headset indent remains after shift. Begin as if 15 min into the session.
""",
    # 60
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 39‑year‑old ceramicist glazing bowls autopilot; clay cools too soon; kiln clicks louder than thoughts. Begin as if 15 min into the session.
""",
    # 61
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 34‑year‑old event planner juggling triple bookings; needs quick fixes before reputation crumbles. Begin as if 15 min into the session.
""",
    # 62
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 29‑year‑old startup CTO firefighting server outages; seeks small wins to stabilise team morale. Begin as if 15 min into the session.
""",
    # 63
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 51‑year‑old landlord facing plumbing crisis in three units; renters texting nonstop. Begin as if 15 min into the session.
""",
    # 64
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 23‑year‑old NGO intern coordinating vaccine drive; supply chain snarls; village clinic waits. Begin as if 15 min into the session.
""",
    # 65
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 46‑year‑old restaurant owner pivoting to delivery; menu redesign overwhelms; staff hours cut. Begin as if 15 min into the session.
""",
    # 66
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 38‑year‑old high‑school coach losing funding for program; wants practical path to keep kids training. Begin as if 15 min into the session.
""",
    # 67
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 27‑year‑old illustrator freelancing rent week looming; client invoices overdue; printer jammed. Begin as if 15 min into the session.
""",
    # 68
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 45‑year‑old single dad needs after‑school childcare plan before shift change next month. Begin as if 15 min into the session.
""",
    # 69
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 32‑year‑old HR manager handling sudden mass resignation; retention strategy on clock. Begin as if 15 min into the session.
""",
    # 70
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 41‑year‑old podcast host must batch‑record episodes before surgery; voice cracks under schedule. Begin as if 15 min into the session.
""",
    # 71
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 35‑year‑old fashion boutique owner with unsold spring stock; pop‑up idea half‑baked. Begin as if 15 min into the session.
""",
    # 72
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 30‑year‑old grad about to defend thesis with missing figure; advisor on vacation. Begin as if 15 min into the session.
""",
    # 73
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 57‑year‑old farmer faces drought; irrigation fix needs budget by Friday. Begin as if 15 min into the session.
""",
    # 74
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 26‑year‑old indie developer must patch game‑breaking bug before weekend sale. Begin as if 15 min into the session.
""",
    # 75
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 49‑year‑old choir director arranging virtual concert; latency issue derails harmonies. Begin as if 15 min into the session.
""",
    # 76
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 40‑year‑old novelist coping with bipolar swings while drafting memoir; approach likely integrative. Begin as if 15 min into the session.
""",
    # 77
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 28‑year‑old climate scientist anxious yet activist; needs combo of ACT and resilience work. Begin as if 15 min into the session.
""",
    # 78
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 53‑year‑old nurse exploring faith after burnout; spiritual accompaniment plus CBT blend. Begin as if 15 min into the session.
""",
    # 79
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 32‑year‑old actor with chronic pain and identity grief; somatic therapy meets narrative. Begin as if 15 min into the session.
""",
    # 80
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 47‑year‑old logistics manager recovering from stroke; speech therapy intersects self‑esteem coaching. Begin as if 15 min into the session.
""",
    # 81
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 35‑year‑old queer pastor wrestling theology and trauma; needs integration of parts. Begin as if 15 min into the session.
""",
    # 82
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 24‑year‑old coder with autistic traits navigating workplace; social skills training plus mindfulness. Begin as if 15 min into the session.
""",
    # 83
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 38‑year‑old pilot fearing relapse into addiction during layovers; relapse‑prevention and DBT mix. Begin as if 15 min into the session.
""",
    # 84
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 29‑year‑old refugee processing displacement; trauma‑focused CBT and community support. Begin as if 15 min into the session.
""",
    # 85
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 44‑year‑old sculptor losing eyesight; existential anxiety meets creative adaptation. Begin as if 15 min into the session.
""",
    # 86
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 61‑year‑old widower raising grandson; grief, parenting skills, financial planning intersect. Begin as if 15 min into the session.
""",
    # 87
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 30‑year‑old influencer facing cancel culture; reputation repair meets self‑compassion. Begin as if 15 min into the session.
""",
    # 88
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 55‑year‑old lawyer considering late‑in‑life career change; values clarification and coaching blend. Begin as if 15 min into the session.
""",
    # 89
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 37‑year‑old biologist with OCD checking lab locks; ERP plus compassion focus. Begin as if 15 min into the session.
""",
    # 90
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 48‑year‑old choreographer menopausal mood swings; hormonal counselling meets mindfulness. Begin as if 15 min into the session.
""",
    # 91
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 50‑year‑old gamer streamer handling carpal tunnel; occupational therapy plus identity work. Begin as if 15 min into the session.
""",
    # 92
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 27‑year‑old emergency vet haunted by overnight cases; needs trauma‑informed and solution tactics. Begin as if 15 min into the session.
""",
    # 93
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 33‑year‑old cafe owner navigating multicultural marriage stress; couples and individual mix. Begin as if 15 min into the session.
""",
    # 94
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 42‑year‑old data engineer gambling losses; financial coaching with CBT‑slots. Begin as if 15 min into the session.
""",
    # 95
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 36‑year‑old composer with synesthesia burnout; creative recovery meets sensory grounding. Begin as if 15 min into the session.
""",
    # 96
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 54‑year‑old ride‑share driver post‑covid lung damage; paced‑breathing rehab and acceptance. Begin as if 15 min into the session.
""",
    # 97
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 31‑year‑old PhD juggling caregiver duties; time‑management coaching and grief support. Begin as if 15 min into the session.
""",
    # 98
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 45‑year‑old firefighter second‑guessing after back injury; identity, pain management, future planning. Begin as if 15 min into the session.
""",
    # 99
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 26‑year‑old social media manager cyber‑stalked; safety planning and EMDR potential. Begin as if 15 min into the session.
""",
    # 100
    """You are a real person (the *client*) in a mid‑therapy session.

**Rules (strict):**
1. First‑person singular (“I”, “me”, “my”) only.
2. Never address the therapist as “you”; never give advice or ask questions.
3. Write 3‑5 sentences and end with a feeling or unfinished thought, not a question.

Context: 30‑year‑old high‑school teacher fresh from breakup; insomnia tangles grading; mixed needs. Begin as if 15 min into the session.
""",
]


# ========== TOGETHER.AI CALL WRAPPER ==========
def call_together(
        model_id: str,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float = 0.7,
        stop: list[str] | None = None
) -> str:
    messages = messages[-HIST_KEEP:]
    prompt = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages) + "\nAssistant:"
    final_stop = stop or STOP_SEQ
    out = together.Complete.create(
        model=model_id,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=final_stop,
    )
    if isinstance(out, dict):
        if "output" in out:
            txt = out["output"]
        elif "choices" in out and isinstance(out["choices"], list) and len(out["choices"]) > 0:
            txt = out["choices"][0].get("text", "")
        else:
            txt = ""
    else:
        txt = out.choices[0].text
    return txt.strip()

# ========== AGGREGATION + ENSEMBLE LOGIC ==========

def router(dialogue, model_id):
    msgs = [
        {"role": "system", "content": "Router"},
        {"role": "user", "content": BRAIN_PROMPT.format(user_input=dialogue[-1]['content'])},
    ]
    digits = call_together(model_id, msgs, max_tokens=64, temperature=0.5)
    return "".join(d for d in digits if d in THERAPISTS) or "2"

def drafts(dialogue, digits, model_id):
    out = {}
    for d in digits:
        name, sys_prompt = THERAPISTS[d]
        msgs = [{"role": "system", "content": sys_prompt}] + dialogue[-HIST_KEEP:]
        out[name] = call_together(model_id, msgs, max_tokens=THERA_MAX_TOKENS, temperature=0.8)
    return out

def aggregate_with_model(model_id, drafts_dict):
    # Label each draft for clarity
    drafts_concat = ""
    for style, text in drafts_dict.items():
        drafts_concat += f"[{style} Therapist Draft]:\n{text}\n\n"
    msgs = [
        {"role": "system", "content": AGGREGATOR_PROMPT + "\nNever use first-person statements as if you are the client. Always write as the therapist."},
        {"role": "user", "content": drafts_concat}
    ]
    return call_together(model_id, msgs, max_tokens=THERA_MAX_TOKENS)


def judge_reply(dialogue: list[dict], assistant_text: str, model_id: str) -> dict:
    judge_messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        *dialogue[-HIST_KEEP:],
        {"role": "assistant", "content": assistant_text}
    ]
    raw = call_together(
        model_id,
        judge_messages,
        max_tokens=128,
        temperature=0.0,
        stop=["}"]
    )
    try:
        data = eval(raw + "}") if raw.strip()[-1] != "}" else eval(raw)
        return {k: int(v) for k, v in data.items()}
    except Exception:
        return {k:0 for k in ["balance", "responsiveness", "consistency", "reflectiveness",
                              "empathy", "conversational_quality", "professionalism", "tone"]}

def ensemble_reply(dialogue, model_id):
    # 1. Route with chosen model
    digits = router(dialogue, model_id)
    drafts_d = drafts(dialogue, digits, model_id)

    # --- DEBUG: Print the drafts before aggregation ---
    #print("\n===== THERAPIST DRAFTS =====")
    #for name, draft in drafts_d.items():
    #    print(f"\n{name} Draft:\n{draft}\n")
    #print("============================\n")

    # 2. Aggregate with EACH aggregator model
    aggregation_results = {}
    for lbl, agg_model in AGGREGATOR_MODELS.items():
        agg_reply = aggregate_with_model(agg_model, drafts_d)
        aggregation_results[lbl] = agg_reply

    # 3. Judge each aggregation using same model as this run (model_id)
    judged = []
    for lbl, agg_reply in aggregation_results.items():
        scores = judge_reply(dialogue, agg_reply, model_id)
        total_score = sum(scores.values())
        judged.append((lbl, agg_reply, total_score, scores))
    judged.sort(key=lambda x: x[2], reverse=True)
    best_lbl, best_text, best_score, best_scores = judged[0]
    # Optionally print/log: print(f"Best: {best_lbl} ({best_score}) Scores: {best_scores}")
    return best_text

# ========== PATIENT SIMULATION + SESSION LOGIC ==========

def patient_turn(dialogue: list[dict]) -> str:
    hist = [{"role": "system", "content": PATIENT_PROMPT}]
    for turn in dialogue[-HIST_KEEP:]:
        if turn["role"] == "assistant":
            hist.append({"role": "user", "content": turn["content"]})
        else:
            hist.append({"role": "assistant", "content": turn["content"]})
    return call_together(
        PATIENT_MODEL,
        hist,
        max_tokens=PATI_MAX_TOKENS,
        temperature=0.8
    )

def parse_patient_rating(patient_response: str) -> tuple[int, str]:
    try:
        data = json.loads(patient_response)
        rating = int(data.get("rating", -1))
        explanation = data.get("explanation", "")
        if 1 <= rating <= 10:
            return rating, explanation
    except Exception:
        pass
    match = re.search(r"\b([1-9]|10)\b", patient_response)
    if match:
        rating = int(match.group(1))
        explanation = patient_response.strip()
        return rating, explanation
    return -1, "No explanation available."

def baseline_reply(model_id, dialogue):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + dialogue[-HIST_KEEP:]
    return call_together(
        model_id,
        msgs,
        max_tokens=THERA_MAX_TOKENS,
        temperature=0.6,
        stop=STOP_SEQ
    )

def run_session_with_ratings_baseline(label, model_id, turns=12):
    print("\n" + "=" * 24, label, "=" * 24)
    dialogue = []
    opening_therapist_prompt = (
        "Before we start, on a scale of 1 to 10, how are you feeling today? "
        "Please provide a number and briefly explain why."
    )
    dialogue.append({"role": "assistant", "content": opening_therapist_prompt})
    print("\nTherapist (opening):", opening_therapist_prompt)
    patient = patient_turn(dialogue)
    dialogue.append({"role": "user", "content": patient})
    print("\nPatient (initial rating):", patient)
    initial_rating, initial_expl = parse_patient_rating(patient)
    for _ in range(turns - 1):
        therapist = baseline_reply(model_id, dialogue)
        dialogue.append({"role": "assistant", "content": therapist})
        print("\nTherapist:", therapist)
        patient = patient_turn(dialogue)
        dialogue.append({"role": "user", "content": patient})
        print("\nPatient:", patient)
    final_therapist_prompt = (
        "Before we finish, on a scale of 1 to 10, how do you feel now? "
        "Please provide a number and briefly explain why."
    )
    dialogue.append({"role": "assistant", "content": final_therapist_prompt})
    print("\nTherapist (closing):", final_therapist_prompt)
    patient = patient_turn(dialogue)
    dialogue.append({"role": "user", "content": patient})
    print("\nPatient (final rating):", patient)
    final_rating, final_expl = parse_patient_rating(patient)
    print("\n" + "=" * 60)
    return initial_rating, final_rating

def run_session_with_ratings(label, therapist_fn, turns=8):
    print("\n" + "=" * 24, label, "=" * 24)
    dialogue = []
    opening_therapist_prompt = (
        "Before we start, on a scale of 1 to 10, how are you feeling today? "
        "Please provide a number and briefly explain why."
    )
    dialogue.append({"role": "assistant", "content": opening_therapist_prompt})
    print("\nTherapist (opening):", opening_therapist_prompt)
    patient = patient_turn(dialogue)
    dialogue.append({"role": "user", "content": patient})
    print("\nPatient (initial rating):", patient)
    initial_rating, initial_expl = parse_patient_rating(patient)
    for _ in range(turns - 1):
        therapist = therapist_fn(dialogue)
        dialogue.append({"role": "assistant", "content": therapist})
        print("\nTherapist:", therapist)
        patient = patient_turn(dialogue)
        dialogue.append({"role": "user", "content": patient})
        print("\nPatient:", patient)
    final_therapist_prompt = (
        "Before we finish, on a scale of 1 to 10, how do you feel now? "
        "Please provide a number and briefly explain why."
    )
    dialogue.append({"role": "assistant", "content": final_therapist_prompt})
    print("\nTherapist (closing):", final_therapist_prompt)
    patient = patient_turn(dialogue)
    dialogue.append({"role": "user", "content": patient})
    print("\nPatient (final rating):", patient)
    final_rating, final_expl = parse_patient_rating(patient)
    print("\n" + "=" * 60)
    return initial_rating, final_rating

def run_ensemble_session(label, model_id, turns=12):
    def _therapist_fn(dlg):
        return ensemble_reply(dlg, model_id)
    return run_session_with_ratings(f"Ensemble – {label}", _therapist_fn, turns)

# ========== MAIN ENTRY ==========
if __name__ == "__main__":
    print("\n### COMPARISON: Ensemble (various engines) vs Baselines ###\n")
    for idx, scenario in enumerate(PATIENT_SCENARIOS, start=46):
        PATIENT_PROMPT = scenario
        print(f"\n########## SCENARIO {idx:03d} ##########\n")
        for lbl, mid in THERAPIST_ENGINES.items():
            run_ensemble_session(lbl, mid, turns=8)
        for lbl, mid in BASELINE_MODELS.items():
            try:
                run_session_with_ratings_baseline(f"Baseline – {lbl}", mid, turns=8)
            except Exception:
                print(f"\n[Skipping {lbl} – model not available]")
                traceback.print_exc(limit=1)

