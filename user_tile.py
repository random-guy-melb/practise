st.markdown("""
<style>
/* ─────────────────────────────────────────────────────────────────────
   1) USER MESSAGE STYLES 
   Right-aligned, glossy orange bubble, shrink-to-fit for short messages
   ───────────────────────────────────────────────────────────────────── */
[data-testid="stChatMessageAvatarUser"] {
    display: none !important;  /* Hide user avatar (optional) */
}

.stChatMessage:has([data-testid="stChatMessageAvatarUser"]) {
    display: inline-flex;       /* shrink bubble to content width */
    align-items: center;        /* vertically center text (esp. for short msgs) */
    margin-left: auto;          /* push bubble to the right */
    margin-right: 1rem;         /* space from the right edge */
    margin-top: 0.5rem;         /* vertical spacing above */
    margin-bottom: 0.5rem;      /* vertical spacing below */
    max-width: 70%;             /* prevent super-wide bubbles */
    vertical-align: top;        
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg, #E86D5C 0%, #F9B3A4 100%);
    color: black;
    border-radius: 10px;
    padding: 12px 16px;         
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    min-height: 50px;           /* ensures short messages aren't too tiny */
}

[data-testid="stChatMessageAvatarUser"] + [data-testid="stChatMessageContent"] {
    text-align: left !important;  /* left-align text inside the user bubble */
}

/* Keep bubble's interior background transparent */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > * {
    background: transparent !important;
}

/* ─────────────────────────────────────────────────────────────────────
   2) ASSISTANT (BOT) MESSAGE STYLES
   Full-width, white-to-light-grey gradient bubble
   ───────────────────────────────────────────────────────────────────── */
.stChatMessage:has([data-testid="stChatMessageAvatarBot"]) {
    display: block;       /* block-level so it can take full width */
    width: 100%;          /* occupy entire container width */
    margin: 0;            /* remove default margins */
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarBot"]) {
    background: linear-gradient(135deg, #FFFFFF 0%, #F7F7F7 100%);
    color: black;
    border-radius: 10px;
    padding: 12px 16px;   
    border: 1px solid rgba(0, 0, 0, 0.05);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
    min-height: 50px;     /* ensures short messages don’t collapse */
    /* If you want some spacing at top/bottom, you can add margin here:
       e.g. margin: 0.5rem 0; */
}

/* Align text inside the bot bubble as desired (usually left) */
[data-testid="stChatMessageAvatarBot"] + [data-testid="stChatMessageContent"] {
    text-align: left !important;
}
</style>
""", unsafe_allow_html=True)
