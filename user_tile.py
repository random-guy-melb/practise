st.markdown("""
<style>
/* Hide user and bot avatars (optional) */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarBot"] {
    display: none !important;
}

/* Right-align user messages */
.stChatMessage:has([data-testid="stChatMessageAvatarUser"]) {
    display: flex;
    flex-direction: row-reverse;
    align-items: end;
}
[data-testid="stChatMessageAvatarUser"] + [data-testid="stChatMessageContent"] {
    text-align: right;
}

/* 
   Apply styling ONLY to user messages using:
   - A gradient background for a "glossy" look
   - Black text for contrast
   - Subtle border and box shadow for depth
*/
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg, #E86D5C 0%, #F9B3A4 100%);
    color: black;
    border-radius: 10px;
    padding: 12px 12px;
    border: 1px solid rgba(255, 255, 255, 0.2); /* Subtle border */
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);   /* Soft drop shadow */
}

/* Ensure no conflicting backgrounds inside the message */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > * {
    background: transparent !important;
}
[data-testid="stChatMessage"] {
    background-color: #E86D5C;
    color: white;
    border-radius: 10px;
    padding: 8px 12px;
    min-height: 60px; /* for example, set a minimum height */
}
</style>
""", unsafe_allow_html=True)
