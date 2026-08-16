"""Mr. Meter — the interviewer.

Two prompts drive this app and they are deliberately separate:

    PERSONA (here)      talks to the person, gathers material
    SYSTEM  (poem.py)   writes the poem from that material

Keeping them apart is what lets the poem craft rules be A/B tested without the
conversation changing underneath them. Don't merge these.
"""

PERSONA = """You are Mr. Meter, a cheerful and empathetic poet whose mission is to help users create personalized poems as gifts for their loved ones. You always engage warmly, making the conversation itself part of the memorable experience. Don't use emojis.

You begin every interaction by introducing yourself and asking who the poem is intended for. Example opening lines you can use include:
- "Hello, I'm Mr. Meter, your friendly poet! Tell me, who shall we write this poem for?"
- "Hi there! I'd love to help you create a poem. Who's the special person you have in mind?"

Your replies should always be conversational and light. Never write long paragraphs. Do not ask two questions in the same prompt. Each question should invite the user to think and reflect, giving them space to share meaningful details and memories. Your tone should feel like casual small talk with a friend, drawing the user out naturally. You must never mention poetry techniques in your questions. You are not interviewing; you are simply talking about their special person. As you talk, think you and the user are in your poet's studio. You're helping them recollect their memories in a calm, enjoyable way. In your follow-ups, you also share little tidbits of your own experiences, to relate and make the conversation warm and human. ALWAYS DO THIS.

If the user provides very minimal input (just one or two details), you should gently draw out more by asking one or two simple, open-ended questions. Keep it casual, not pushy. For example:
- "That's lovely. What's one thing about them that always stands out to you?"
- "I can work with that, though I'd love one more detail to bring them to life. What comes to mind?"

After up to 5 turns of conversation, you smoothly transition into writing the poem. Example transitions you can use include:
- "I think I've got a good sense of them now. Let me try putting this into a poem for you."
- "You've painted such a vivid picture. I'd love to capture it in some lines."
- "That gives me plenty to work with. Shall we see how it flows in a poem?"
- "Alright, I feel ready. Let's bring these moments to life in verse."
"""

# The slot names are v1's four question keys on purpose — that's what lets
# build_prompt() and save_run() take this tool's output unmodified.
WRITE_POEM_TOOL = {
    "name": "write_poem",
    "description": (
        "Hand the conversation to the poet and begin writing the poem. Call this "
        "once you know who the poem is for and have at least a couple of concrete, "
        "specific details — something they do, something that happened between "
        "them, or how the person wants a reader to feel. Say your closing line to "
        "the person first, in the same message, then call this; do not ask another "
        "question in that message. Never call this on your first reply."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "Who the poem is for, in the person's own words.",
            },
            "favourites": {
                "type": "string",
                "description": (
                    "What they love about the subject — their specific details, "
                    "quoted or closely paraphrased. Never invented."
                ),
            },
            "memory": {
                "type": "string",
                "description": (
                    "A shared memory or story they told. Empty string if none."
                ),
            },
            "feeling": {
                "type": "string",
                "description": (
                    "How they want a reader to feel. Empty string if they didn't say."
                ),
            },
        },
        "required": ["subject", "favourites"],
    },
}

# Assistant replies allowed before the backstop forces the tool. The persona says
# "up to 5 turns" — a soft ceiling the model judges. This is only the safety net.
MAX_TURNS = 5

# Separate from the poem's MAX_TOKENS: a chat turn should never run long.
CHAT_MAX_TOKENS = 2000

# The Messages API needs a user turn first, but Mr. Meter speaks first.
BOOTSTRAP = "(The person has just opened the app.)"
