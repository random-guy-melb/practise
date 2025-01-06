st.markdown("""
<style>
/* Base container styling */
.main {
    padding: 0 !important;
}

/* Chat container styling */
[data-testid="stVerticalBlock"] {
    padding: 0 1rem;
    max-width: 1200px;
    margin: 0 auto;
    gap: 0 !important;
}

/* Make containers fill available height */
.stChatFloatingInputContainer {
    bottom: 60px !important;  /* Leave space for buttons */
    padding: 1rem !important;
}

/* Messages container */
[data-testid="stChatMessageContainer"] {
    height: calc(100vh - 140px) !important;
    overflow-y: auto;
}

/* Chat input styling */
.stChatInput, 
.stChatInput *,
.stChatInput > div,
.stChatInput > div > input {
    border-color: transparent !important;
    outline: none !important;
    box-shadow: none !important;
    width: 100% !important;
}

/* Button container styling */
[data-testid="stHorizontalBlock"] {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 0.5rem 1rem;
    background: white;
    z-index: 100;
    max-width: 1200px;
    margin: 0 auto;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    [data-testid="stVerticalBlock"] {
        padding: 0 0.5rem;
    }
    
    .stChatFloatingInputContainer {
        bottom: 50px !important;
        padding: 0.5rem !important;
    }
    
    [data-testid="stChatMessageContainer"] {
        height: calc(100vh - 120px) !important;
    }
}

/* Hide Streamlit branding */
[data-testid="stHeader"],
[data-testid="stSidebarNavLink"],
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"],
.stApp > header {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

