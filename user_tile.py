st.markdown("""
<style>
/* Hide the user avatar (optional) */
[data-testid="stChatMessageAvatarUser"] {
    display: none !important;
}

/*
  Make user messages an inline-flex container:
  - margin-left: auto pushes them to the right
  - margin-right: gives space from the right edge
  - align-items: center vertically centers short text
  - max-width limits how wide messages can get
*/
.stChatMessage:has([data-testid="stChatMessageAvatarUser"]) {
    display: inline-flex; 
    align-items: center;       /* vertically center text/content */
    margin-left: auto; 
    margin-right: 1rem; 
    max-width: 70%;
    vertical-align: top;       /* ensures they're aligned along top row */
}

/*
  Style only the user’s bubble:
  - gradient background
  - padding for comfortable interior space
  - min-height ensures short messages look balanced
*/
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg, #E86D5C 0%, #F9B3A4 100%);
    color: black;
    border-radius: 10px;
    padding: 12px 16px; 
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    min-height: 50px;          /* Adjust so small messages aren't too short */
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
