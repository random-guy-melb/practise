
st.markdown("""
<style>
/* Hide the user avatar (optional) */
[data-testid="stChatMessageAvatarUser"] {
    display: none !important;
}

/* Hide the assistant avatar */
[data-testid="stChatMessageAvatarAssistant"] {
    display: none !important;
}

/* User message container styling */
.stChatMessage:has([data-testid="stChatMessageAvatarUser"]) {
    display: inline-flex; 
    align-items: center;       
    margin-left: auto; 
    margin-right: 1rem; 
    max-width: 70%;
    vertical-align: top;       
}

/* User message bubble styling */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg, #E86D5C 0%, #F9B3A4 100%);
    color: black;
    border-radius: 10px;
    padding: 12px 16px; 
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    min-height: 50px;          
}

/* Assistant message styling - with rounded edges */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: #f7f7f8;
    border-radius: 10px;
    margin: 8px 0;
    padding: 12px 16px;
    width: 100%;
}

/* Text alignment and transparent backgrounds */
[data-testid="stChatMessageAvatarUser"] + [data-testid="stChatMessageContent"] {
    text-align: left !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > * {
    background: transparent !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) > * {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)
