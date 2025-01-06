from userdata import DatabaseManager
import streamlit as st
import os, sys, time, pickle, hashlib, json

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

st.set_page_config(
    page_title="Login",
    page_icon="🔐",
    layout="centered",
)

st.markdown("""
    <style>
    /* Base styles */
    body {
        background-color: #fff8e7 !important;
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
        overflow-x: hidden;
    }

    /* Hide Streamlit header elements */
    header[data-testid="stHeader"],
    .stApp > header {
        display: none !important;
    }

    /* Remove top padding from Streamlit container */
    .main .block-container {
        margin-top: 0 !important;
    }

    /* Left panel - using viewport units and flexbox */
    .left-panel {
        position: fixed;
        left: 0;
        top: 0;
        bottom: 0;
        width: 60%;
        background-color: #fff8e7;
        padding: clamp(1rem, 3vw, 2rem);
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        padding-top: min(20%, 150px);
    }

    /* Right panel - using viewport units */
    .block-container {
        position: fixed !important;
        top: 0 !important;
        bottom: 0 !important;
        left: 60% !important;
        width: 40% !important;
        padding: clamp(1rem, 3vw, 3rem) !important;
        background-color: #ffffff;
        border-radius: 8px 0 0 0;
        box-shadow: -3px 0 12px rgba(0, 0, 0, 0.08);
        overflow-y: auto !important;
        margin: 0 !important;
    }

    /* Responsive typography */
    .title-text {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: clamp(2rem, 4vw, 4rem) !important;
        line-height: 1.1 !important;
        text-align: left !important;
        margin-left: clamp(1rem, 2vw, 2rem) !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: -0.03em !important;
    } 

    .title-text-secondary-title {
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: clamp(1.2rem, 2.2vw, 2.2rem) !important;
        line-height: 1.1 !important;
        text-align: left !important;
        margin-left: clamp(1rem, 2vw, 2rem) !important;
        margin-top: clamp(1rem, 1vw, 1rem) !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: -0.02em !important;
    } 

    .tagline-text {
        color: #FF8C00 !important;
        font-weight: 600 !important;
        font-size: clamp(1.2rem, 2.2vw, 2.2rem) !important;
        line-height: 1.3 !important;
        text-align: left !important;
        margin-left: clamp(1rem, 2vw, 2rem) !important;
        margin-top: clamp(1rem, 1vw, 1rem) !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: -0.02em !important;
    }

    /* Typography styles */
    h1 {
        font-size: clamp(1.5rem, 2vw, 2rem);
        margin-bottom: 0.5rem;
        color: #2c2c2c;
        text-align: center;
        font-weight: 700;
    }

    h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        color: #2c2c2c;
    }

    p, div {
        color: #444 !important;
    }

    /* Tabs styling */
    div[role=radiogroup] {
        border-bottom: 1px solid #e3e3e3;
        margin-bottom: 1.75rem;
        display: flex;
        justify-content: center;
        padding-top: 3rem;

    }

    div[role=radiogroup] label {
        font-size: clamp(0.9rem, 1.05vw, 1.05rem);
        border: none !important;
        font-weight: 400;
        color: #777 !important;
        padding: 0.75rem 1.25rem !important;
        margin: 0 0.5rem !important;
        border-radius: 0 !important;
        cursor: pointer;
    }

    div[role=radiogroup] label[data-selected="true"] {
        color: #000 !important;
        border-bottom: 2px solid #10a37f !important;
        font-weight: 700 !important;
    }

    /* Input fields */

    input .stTextInput > div {
        font-size: clamp(0.9rem, 1vw, 1rem) !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.7rem !important;
        background-color: #F3F4F6 !important;
        box-shadow: none !important;
        white-space: normal !important; /* Allow text wrapping */
        overflow: visible !important; /* Ensure no hidden overflow */
        text-overflow: clip !important; /* Prevent text ellipsis */

    }

    /* Password fields */
    [type="password"] {
        background-color: #F3F4F6 !important;
        border: none !important;
        outline: none !important;
    }

    /* Security Question Dropdown */
    .stSelectbox > div[data-baseweb="select"],
    .stSelectbox > div[data-baseweb="select"]:focus,
    .stSelectbox > div[data-baseweb="select"]:hover,
    .stSelectbox > div[data-baseweb="select"]:active,
    .stSelectbox > div[data-baseweb="select"][data-focusvisible="true"] {
        background-color: #F3F4F6 !important;
        border: none !important;
        border-radius: 8px !important;
        color: #666 !important;
        -webkit-autofill: none !important;
        -webkit-box-shadow: 0 0 0 30px #F3F4F6 inset !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* Prevent autofill suggestions */
    .stSelectbox input,
    .stSelectbox select,
    .stSelectbox [role="combobox"] {
        autocomplete: "new-password" !important;
        -webkit-autofill: none !important;
        -webkit-text-fill-color: #666 !important;
        transition: background-color 5000s ease-in-out 0s !important;
        background-color: #F3F4F6 !important;
        caret-color: transparent !important;
    }

    /* Hide autofill icon */
    .stSelectbox input::-webkit-contacts-auto-fill-button,
    .stSelectbox input::-webkit-credentials-auto-fill-button,
    .stSelectbox [role="combobox"]::-webkit-contacts-auto-fill-button,
    .stSelectbox [role="combobox"]::-webkit-credentials-auto-fill-button {
        visibility: hidden;
        display: none !important;
        pointer-events: none;
        position: absolute;
        right: 0;
    }

    /* Remove all possible border and outline states */
    .stSelectbox > div > div,
    .stSelectbox div[data-baseweb],
    .stSelectbox div[data-baseweb] *,
    .stSelectbox [role="combobox"],
    .stSelectbox [role="combobox"]:focus,
    .stSelectbox [role="combobox"]:hover,
    .stSelectbox > div > div[data-testid],
    .stSelectbox [data-focusvisible="true"],
    .stSelectbox [data-focused="true"] {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* Tabs styling with stronger selectors */
    div[role=radiogroup] label {
        font-size: clamp(0.9rem, 1.05vw, 1.05rem) !important;
        border: none !important;
        font-weight: 400 !important;
        color: #777 !important;
        padding: 0.75rem 1.25rem !important;
        margin: 0 0.5rem !important;
        border-radius: 0 !important;
        cursor: pointer;
    }

    div[role=radiogroup] label[data-selected="true"],
    div[role=radiogroup] label[data-selected="true"] p {
        color: #000 !important;
        border-bottom: 2px solid #10a37f !important;
        font-weight: 700 !important;
    }

    div[role=radiogroup] label p {
        font-weight: inherit !important;
    }

    /* Disable autofill styles */
    .stSelectbox select:-webkit-autofill,
    .stSelectbox select:-webkit-autofill:hover,
    .stSelectbox select:-webkit-autofill:focus,
    .stSelectbox select:-webkit-autofill:active {
        -webkit-box-shadow: 0 0 0 30px #F3F4F6 inset !important;
        -webkit-text-fill-color: #666 !important;
        transition: background-color 5000s ease-in-out 0s;
        background-color: #F3F4F6 !important;
    }

    .stSelectbox [data-baseweb="select"] div[data-testid="stMarkdownContainer"] {
        color: #666 !important;
        -webkit-autofill: none !important;
    }

    /* Style dropdown options */
    .stSelectbox [role="listbox"] {
        background-color: white !important;
        border: 1px solid #eee !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }

    .stSelectbox [role="option"] {
        padding: 8px 16px !important;
        color: #666 !important;
    }

    .stSelectbox [role="option"]:hover {
        background-color: #F3F4F6 !important;
    }

    /* Password visibility toggle */
    button[aria-label="Toggle password visibility"] {
        cursor: pointer !important;
        opacity: 0.7 !important;
        padding: 8px !important;
        background: none !important;
        border: none !important;
        color: #666 !important;
    }

    button[aria-label="Toggle password visibility"]:hover {
        opacity: 1 !important;
    }

    /* Password input container */
    .stTextInput > div[data-baseweb="input"] {
        display: flex !important;
        align-items: center !important;
    }

    /* Password reveal icon container */
    .stTextInput > div[data-baseweb="input"] > div:last-child {
        display: flex !important;
        align-items: center !important;
        padding-right: 8px !important;
    }

    /* Input field containers */
    .stTextInput > div[data-baseweb="input"],
    .stTextInput > div[data-baseweb="input"]:hover,
    .stTextInput > div[data-baseweb="input"]:focus,
    .stTextInput > div[data-baseweb="input"]:active {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        background-color: #F3F4F6 !important;
        border-radius: 8px !important;
        display: flex !important;
        align-items: center !important;
    }

    /* Input fields themselves */
    .stTextInput input,
    .stTextInput input:focus,
    .stTextInput input:hover,
    .stTextInput input:active {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        background-color: transparent !important;
    }

    /* Ensure proper spacing for password fields */
    .stTextInput [type="password"] {
        padding-right: 40px !important;
    }

    /* Buttons */
    .stButton button {
        background-color: #10a37f !important;
        border: none;
        border-radius: 5px;
        color: #fff;
        font-size: clamp(0.9rem, 1vw, 1rem);
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        padding: 0.7rem 1.5rem;
        margin-top: 0.7rem;
        cursor: pointer;
    }

    .stButton button:hover {
        background-color: #0e846b !important;
    }

    /* Typing animation */
    .typing-container {
        display: inline-block;
        max-width: 100%;
    }

    .typing-text {
        display: inline-block;
        overflow: hidden;
        white-space: nowrap;
        border-right: .15em solid #FF8C00;
        margin: 0;
        width: 0;
        animation: 
            typing 2s steps(30, end) forwards,
            blink-caret .75s step-end infinite;
    }

    @keyframes typing {
        from { width: 0 }
        to { 
            width: 100%;
            opacity: 1;
        }
    }

    @keyframes blink-caret {
        from, to { border-color: transparent }
        50% { border-color: #FF8C00; }
    }

    /* Media Queries */
    @media screen and (max-width: 1200px) {
        .left-panel {
            padding-top: 15%;
        }
    }

    @media screen and (max-width: 768px) {
        .left-panel {
            width: 50% !important;
            padding-top: 10%;
        }

        .block-container {
            left: 50% !important;
            width: 50% !important;
        }

        .title-text {
            font-size: clamp(1.8rem, 3vw, 2.5rem) !important;
        }

        .title-text-secondary-title {
            font-size: clamp(1rem, 1.8vw, 1.5rem) !important;
        }

        .tagline-text {
            font-size: clamp(1rem, 1.8vw, 1.5rem) !important;
        }
    }

    @media screen and (max-width: 480px) {
        .left-panel {
            width: 45% !important;
            padding-top: 5%;
        }

        .block-container {
            left: 45% !important;
            width: 55% !important;
        }

        .title-text {
            font-size: clamp(1.2rem, 4vw, 1.8rem) !important;
            margin-left: clamp(0.5rem, 1vw, 1rem) !important;
        }

        title-text-secondary-title {
            font-size: clamp(0.8rem, 3vw, 1rem) !important;
            margin-left: clamp(0.5rem, 1vw, 1rem) !important;
        }

        .tagline-text {
            font-size: clamp(0.8rem, 3vw, 1rem) !important;
            margin-left: clamp(0.5rem, 1vw, 1rem) !important;
        }

        .typing-text {
            white-space: normal;
            word-wrap: break-word;
        }
    }



    </style>

    <!-- Left panel with static text -->
    <div class="left-panel">
        <div class="title-text">
            ServiceNow IntelliDefect
        </div>
        <div class="title-text-secondary-title">
            AI-powered Defect Management
        </div>
        <div class="tagline-text">
            <div class="typing-container">
                <span class="typing-text" style="color: #FF8C00;">Making defect resolution simple and efficient</span>
            </div>
        </div>
    </div>

    <script>
    document.addEventListener('DOMContentLoaded', (event) => {
        const textArray = [
            "Welcome to Slack Defects Management System!",
            "This system helps teams track and manage software defects efficiently.",
            "Collaborate, prioritize, and resolve issues seamlessly.",
            "Built with modern technology for optimal performance."
        ];
        let arrayIndex = 0;
        let charIndex = 0;
        let currentText = "";
        let isDeleting = false;
        let pauseEnd = 1000;

        function type() {
            const typedElement = document.getElementById("typed-text");
            if (!typedElement) return;

            currentText = textArray[arrayIndex].substring(
                0, 
                isDeleting ? (charIndex - 1) : (charIndex + 1)
            );
            typedElement.innerText = currentText;

            if (!isDeleting) {
                charIndex++;
                if (charIndex === textArray[arrayIndex].length) {
                    setTimeout(() => { isDeleting = true; }, pauseEnd);
                }
            } else {
                charIndex--;
                if (charIndex === 0) {
                    isDeleting = false;
                    arrayIndex++;
                    if (arrayIndex === textArray.length) {
                        arrayIndex = 0;
                    }
                }
            }

            const speed = isDeleting ? 30 : 60;
            setTimeout(type, speed);
        }

        type();
    });
    </script>
    """, unsafe_allow_html=True)

script_name = os.path.basename(__file__)


def init_session_state():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'email' not in st.session_state:
        st.session_state.email = None
    if 'reset_stage' not in st.session_state:
        st.session_state.reset_stage = 'email'
    if 'temp_email' not in st.session_state:
        st.session_state.temp_email = None
    if 'temp_user_id' not in st.session_state:
        st.session_state.temp_user_id = None
    if "clear_chat_trigger" not in st.session_state:
        st.session_state.clear_chat_trigger = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "reset_flag" not in st.session_state:
        st.session_state.reset_flag = False
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False
    if "last_response_index" not in st.session_state:
        st.session_state.last_response_index = -1


def tabs(tab_list, key="default"):
    tab_key = f"TabGroup_{key}"
    if tab_key not in st.session_state:
        st.session_state[tab_key] = 0

    current_tab_idx = st.radio("Navigation tabs",
                               options=range(len(tab_list)),
                               format_func=lambda x: tab_list[x],
                               key=tab_key,
                               horizontal=True,
                               label_visibility="collapsed")

    st.markdown(f"""
        <style type="text/css">
            div[role=radiogroup] {{
                border-bottom: 2px solid rgba(49, 51, 63, 0.1);
                margin-bottom: 20px;
            }}
            div[role=radiogroup] > label > div:first-of-type {{
               display: none
            }}
            div[role=radiogroup] {{
                flex-direction: unset;
            }}
            div[role=radiogroup] label {{
                padding-bottom: 0.5em;
                border-radius: 0;
                position: relative;
                top: 3px;
                margin-right: 20px;
            }}
            div[role=radiogroup] label .st-fc {{
                padding-left: 0;
            }}
            div[role=radiogroup] label:hover p {{
                color: red;
            }}
            div[role=radiogroup] label:nth-child({current_tab_idx + 1}) {{    
                border-bottom: 2px solid rgb(255, 75, 75);
            }}     
            div[role=radiogroup] label:nth-child({current_tab_idx + 1}) p {{    
                color: rgb(255, 75, 75);
                padding-right: 0;
            }}            
        </style>
    """, unsafe_allow_html=True)

    return current_tab_idx


def handle_email_submit():
    email = st.session_state.reset_email
    question = db.get_security_question(email)
    if question:
        st.session_state.temp_email = email
        st.session_state.reset_stage = 'security'
    else:
        st.error("Email not found")


def handle_security_submit():
    user_id = db.verify_security_answer(st.session_state.temp_email, st.session_state.security_answer)
    if user_id:
        st.session_state.temp_user_id = user_id
        st.session_state.reset_stage = 'new_password'
    else:
        st.error("Incorrect answer")


def handle_password_submit():
    if st.session_state.new_password != st.session_state.confirm_password:
        st.error("Passwords do not match")
        return
    if len(st.session_state.new_password) < 8:
        st.error("Password must be at least 8 characters long")
        return

    db.update_password(st.session_state.temp_user_id, st.session_state.new_password)
    st.success("Password reset successful! Please login.")

    st.session_state.reset_stage = 'email'
    st.session_state.temp_email = None
    st.session_state.temp_user_id = None

    import time
    time.sleep(1)
    st.switch_page(script_name)


def is_session_valid(saved_session):
    if 'timestamp' in saved_session:
        session_age = time.time() - saved_session['timestamp']
        return session_age < (24 * 60 * 60)  # 24 hours
    return False


def generate_device_fingerprint():
    import user_agents
    try:
        # Get request headers from Streamlit
        headers = st.context.headers.to_dict()
        ua_string = headers.get('User-Agent', '')
        user_agent = user_agents.parse(ua_string)

        # Collect fingerprint data
        fingerprint_data = {
            'browser': user_agent.browser.family,
            'browser_version': user_agent.browser.version_string,
            'os': user_agent.os.family,
            'os_version': user_agent.os.version_string,
            'device': user_agent.device.family,
            'ip': headers.get('X-Forwarded-For', headers.get('Remote-Addr', '')),
            'accept_language': headers.get('Accept-Language', ''),
            'screen_info': headers.get('Sec-CH-UA', ''),
            # Additional fingerprint data from available headers
            'cookie_id': headers.get('Cookie', '').split('ajs_anonymous_id=')[1].split(';')[
                0] if 'ajs_anonymous_id=' in headers.get('Cookie', '') else '',
            'host': headers.get('Host', ''),
            'origin': headers.get('Origin', '')
        }

        # Create a unique hash
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_data, sort_keys=True).encode()
        ).hexdigest()

        return fingerprint
    except Exception as e:
        print(f"Error generating fingerprint: {e}")
        return None


def get_last_valid_session():
    """Check for any valid existing session matching the current device"""
    try:
        current_fingerprint = generate_device_fingerprint()
        if not current_fingerprint:
            return None

        sessions_dir = '.streamlit/sessions'
        if os.path.exists(sessions_dir):
            for filename in os.listdir(sessions_dir):
                if filename.endswith('.pkl'):
                    filepath = os.path.join(sessions_dir, filename)
                    with open(filepath, 'rb') as f:
                        saved_session = pickle.load(f)
                        if (is_session_valid(saved_session) and
                                saved_session.get('device_fingerprint') == current_fingerprint):
                            return saved_session
    except Exception as e:
        print(f"Error checking sessions: {e}")
    return None


def handle_login(email, password, db):
    user_id = db.verify_user(email, password)
    if user_id:
        device_fingerprint = generate_device_fingerprint()
        if not device_fingerprint:
            st.error("Could not verify device. Please try again.")
            return

        st.session_state.authenticated = True
        st.session_state.user_id = user_id

        # Save session data with device fingerprint
        session_file = f'.streamlit/sessions/user_{user_id}_{device_fingerprint[:8]}.pkl'
        os.makedirs(os.path.dirname(session_file), exist_ok=True)

        session_data = {
            'authenticated': True,
            'user_id': user_id,
            'email': email,
            'device_fingerprint': device_fingerprint,
            'timestamp': time.time()
        }

        with open(session_file, 'wb') as f:
            pickle.dump(session_data, f)

        st.rerun()
    else:
        st.error("Invalid email or password")


def cleanup_old_sessions():
    """Cleanup expired and duplicate sessions"""
    try:
        sessions_dir = '.streamlit/sessions'
        if os.path.exists(sessions_dir):
            # Group sessions by user_id
            user_sessions = {}
            current_time = time.time()

            for filename in os.listdir(sessions_dir):
                if filename.endswith('.pkl'):
                    filepath = os.path.join(sessions_dir, filename)
                    try:
                        with open(filepath, 'rb') as f:
                            session_data = pickle.load(f)
                            user_id = session_data.get('user_id')

                            # Remove expired sessions
                            if not is_session_valid(session_data):
                                os.remove(filepath)
                                continue

                            if user_id:
                                if user_id not in user_sessions:
                                    user_sessions[user_id] = []
                                user_sessions[user_id].append((filepath, session_data))
                    except:
                        # Remove corrupted files
                        os.remove(filepath)

            # Keep only the most recent session per device for each user
            for user_id, sessions in user_sessions.items():
                device_sessions = {}
                for filepath, session_data in sessions:
                    device_fp = session_data.get('device_fingerprint')
                    if device_fp:
                        if device_fp not in device_sessions or \
                                session_data['timestamp'] > device_sessions[device_fp][1]['timestamp']:
                            if device_fp in device_sessions:
                                os.remove(device_sessions[device_fp][0])  # Remove older session file
                            device_sessions[device_fp] = (filepath, session_data)

    except Exception as e:
        print(f"Error cleaning up sessions: {e}")


@st.fragment
def login_page():

    st.title("Get started", anchor=False)

    SECURITY_QUESTIONS = [
        "What was the name of your first pet?",
        "In which city were you born?",
        "What was your mother's maiden name?",
        "What was the name of your first school?",
        "What was your childhood nickname?"
    ]

    tab_options = ["Login", "Register", "Reset Password"]
    active_tab_idx = tabs(tab_options, "main_tabs")

    if active_tab_idx == 0:  # Login
        email = st.text_input("**Email**", key="login_email")
        password = st.text_input("**Password**", type="password", key="login_password")

        if st.button("**Login**"):
            user_id = db.verify_user(email, password)
            if user_id:
                st.session_state.authenticated = True
                st.session_state.user_id = user_id
                # show_login_progress()
                st.rerun()

            else:
                st.error("Invalid email or password")

    elif active_tab_idx == 1:  # Register
        email = st.text_input("**Email**", key="register_email")
        password = st.text_input("**Password**", type="password", key="register_password")
        confirm_password = st.text_input("**Confirm Password**", type="password")

        security_question = st.selectbox(
            "**Security Question**",
            options=SECURITY_QUESTIONS,
            key="register_security_question",
        )
        security_answer = st.text_input("**Security Answer**", key="register_security_answer")

        if st.button("**Register**"):
            if password != confirm_password:
                st.error("Passwords do not match")
            elif len(password) < 8:
                st.error("Password must be at least 8 characters long")
            elif not security_answer:
                st.error("Security answer is required")
            else:
                if db.create_user(email, password, security_question, security_answer):
                    st.success("Registration successful! Please login.")
                    import time
                    time.sleep(1)
                    st.switch_page(script_name)
                else:
                    st.error("Email already registered")

    else:  # Reset Password
        if st.session_state.reset_stage == 'email':
            st.text_input("**Enter your email**", key="reset_email", on_change=handle_email_submit)

        elif st.session_state.reset_stage == 'security':
            st.info(f"Email: {st.session_state.temp_email}")
            question = db.get_security_question(st.session_state.temp_email)
            st.write("Security Question:", question)
            st.text_input("**Your Answer**", key="security_answer", on_change=handle_security_submit)

        elif st.session_state.reset_stage == 'new_password':
            st.text_input("**New Password**", type="password", key="new_password")
            st.text_input("**Confirm New Password**", type="password", key="confirm_password")
            if st.button("**Reset Password**"):
                handle_password_submit()


@st.fragment
def show_login_progress():
    with st.status(label="Authenticating...", expanded=True) as status:
        import time
        time.sleep(5)
    status.update(
        label="Welcome! You are logged in.", state="complete", expanded=True
    )
    time.sleep(3)


def switch_page():
    st.switch_page("pages/chatbot_test.py")


def main():
    init_session_state()

    if not st.session_state.authenticated:
        saved_session = get_last_valid_session()
        if saved_session and saved_session.get('user_id'):
            st.session_state.authenticated = True
            st.session_state.user_id = saved_session['user_id']
            st.switch_page("pages/chatbot_test.py")

    if not st.session_state.authenticated:
        login_page()
    else:
        switch_page()


if __name__ == "__main__":
    # Initialize database
    db = DatabaseManager()
    main()
