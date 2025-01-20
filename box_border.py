st.markdown("""
<style>
    /* Base message styling */
    [data-testid="stChatMessageContent"] {
        font-family: -apple-system, BlinkMacSystemFont, 
                     "Segoe UI", Roboto, Helvetica, 
                     Arial, sans-serif;
        font-size: 16px;
        padding: 12px 16px;
        border-radius: 20px;
        max-width: 80%;
        line-height: 1.4;
    }
    
    /* Bot message styling */
    .assistant-message [data-testid="stChatMessageContent"] {
        background-color: white;
        color: #1a1a1a;
        border: 1px solid #e5e7eb;
        margin-left: 48px;  /* Space for bot avatar */
    }
    
    /* User message styling */
    .user-message [data-testid="stChatMessageContent"] {
        background-color: #3B82F6;
        color: white;
        margin-left: auto;  /* Align to right */
        margin-right: 1rem;
    }

    /* Hide user avatar */
    .user-message [data-testid="stImage"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)
