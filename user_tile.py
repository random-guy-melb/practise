st.markdown("""
<style>
/* -------------------------------------
   1) BOT AVATAR (Assistant) STYLING
   ------------------------------------- */

/* Show a small circular avatar for the bot.
   Replace the URL below with any image you prefer. */
[data-testid="stChatMessageAvatarBot"] {
    width: 32px;           /* small width */
    height: 32px;          /* small height */
    background-image: url("https://cdn-icons-png.flaticon.com/512/4712/4712104.png");
    background-size: cover;
    background-position: center;
    border-radius: 50%;    /* circular */
    margin-right: 0.5rem;  /* minimal space between avatar & text */
    margin-left: 0;        /* no extra space on left side */
    overflow: hidden;      /* ensure no overflow if image is bigger */
}

/* Hide the default robot icon inside the avatar if Streamlit inserts an <svg>. */
[data-testid="stChatMessageAvatarBot"] > div > svg {
    display: none !important;
}

/* Keep the bot message left-aligned using inline-flex
   so the avatar and text are on a single row. */
.stChatMessage:has([data-testid="stChatMessageAvatarBot"]) {
    display: inline-flex; 
    align-items: center;      /* vertically center the text alongside avatar */
    margin-top: 0.75rem;      /* optional vertical spacing above */
    margin-bottom: 0.75rem;   /* optional vertical spacing below */
    max-width: 70%;           /* limit line length for bot messages */
    vertical-align: top;
}

/* -------------------------------------
   2) USER MESSAGE STYLING (example)
   ------------------------------------- */

/* Hide user avatar */
[data-testid="stChatMessageAvatarUser"] {
    display: none !important;
}

/* Right-align user messages, shrink to content */
.stChatMessage:has([data-testid="stChatMessageAvatarUser"]) {
    display: inline-flex; 
    align-items: center;       /* vertically center short text */
    margin-left: auto; 
    margin-right: 1rem; 
    max-width: 70%;
    vertical-align: top;
}

/* Give user messages a glossy gradient, min-height for short texts */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg, #E86D5C 0%, #F9B3A4 100%);
    color: black;
    border-radius: 10px;
    padding: 12px 16px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    min-height: 50px;
}

/* Left-align text within the user bubble (optional) */
[data-testid="stChatMessageAvatarUser"] + [data-testid="stChatMessageContent"] {
    text-align: left !important;
}

/* Keep bubble's interior background transparent */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > * {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)
