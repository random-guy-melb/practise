st.markdown("""
<style>
/* Hide avatars */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    display: none !important;
}

/* User message container */
.stChatMessage:has([data-testid="stChatMessageAvatarUser"]) {
    display: inline-flex;
    align-items: center;
    max-width: 70%;
    vertical-align: top;
    margin-left: auto;
    margin-right: 1rem;
    margin-bottom: 1rem;
}

/* Assistant message container */
.stChatMessage:has([data-testid="stChatMessageAvatarAssistant"]) {
    display: block;
    margin-right: 2rem;
    margin-left: 2rem;
    margin-bottom: 1rem;
    vertical-align: top;
}

/* User message bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg, #E86D5C 0%, #F9B3A4 100%);
    color: black;
    border-radius: 15px;
    padding: 16px 20px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    min-height: 50px;
}

/* Assistant message bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: white;
    color: black;
    border-radius: 15px;
    padding: 16px 20px;
    border: 1px solid #e5e5e5;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
    min-height: 50px;
    width: calc(100% - 4rem);
}

/* Text alignment within bubbles */
[data-testid="stChatMessageAvatarUser"] + [data-testid="stChatMessageContent"],
[data-testid="stChatMessageAvatarAssistant"] + [data-testid="stChatMessageContent"] {
    text-align: left !important;
}

/* Keep bubble interiors transparent */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > *,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) > * {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)
