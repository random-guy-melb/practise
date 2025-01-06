
import streamlit as st
from streamlit_feedback import streamlit_feedback
import time
import random
from concurrent.futures import ThreadPoolExecutor
import datetime
import pickle, os, user_agents, hashlib, json

# Must be first Streamlit command
st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",

)

# Custom CSS to remove borders and outlines from inputs
st.markdown("""
<style>
/* Chat input specific styling */
.stChatInput, 
.stChatInput *,
.stChatInput > div,
.stChatInput > div > input {
    border-color: transparent !important;
    outline: none !important;
    box-shadow: none !important;
}

.stChatInput > div:focus-within {
    border-color: transparent !important;
    outline: none !important;
    box-shadow: none !important;
}

/* General input styling */
input {
    font-size: 1rem !important; 
    border-radius: 6px !important;
    padding: 0.7rem !important;
}

input:focus,
input:hover,
input:active,
input *,
.stTextInput > div,
[data-baseweb="input"],
[data-baseweb="input"] * {
    border-color: transparent !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Force remove any red borders */
div[data-testid="stChatInput"] {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

[data-testid="stSidebarNavLink"] {
            display: none;
}

[data-testid="stChatMessageAvatarUser"] {
            display: none;
}

[data-testid="stChatMessageAvatarAssistant"] {
            display: none;
}

/* Hide Streamlit header elements */
header[data-testid="stHeader"],
.stApp > header {
    display: none !important;
}

</style>
""", unsafe_allow_html=True)

# Check authentication
if 'authenticated' not in st.session_state or not st.session_state.authenticated:
    st.switch_page("main_test.py")


from models.model import load_model
from snow_app import create_dashboard, process_query, create_sample_data

# Login to Hugging Face
# login(token="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")

def generate_device_fingerprint():

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
def is_session_valid(saved_session):
    if 'timestamp' in saved_session:
        session_age = time.time() - saved_session['timestamp']
        return session_age < (24 * 60 * 60)  # 24 hours
    return False

def get_session_file_path(user_id, device_fingerprint):
    return f'.streamlit/sessions/user_{user_id}_{device_fingerprint[:8]}.pkl'

def save_session_state():
    """Save session state data for specific user"""
    if not st.session_state.get('user_id'):
        return

    device_fingerprint = generate_device_fingerprint()
    session_file = get_session_file_path(st.session_state.user_id, device_fingerprint)
    os.makedirs(os.path.dirname(session_file), exist_ok=True)

    state_to_save = {
        'authenticated': st.session_state.authenticated,
        'device_fingerprint': device_fingerprint,
        'user_id': st.session_state.user_id,
        'messages': st.session_state.get('messages', []),
        'widget_counter': st.session_state.get('widget_counter', 0),
        'clear_chat_trigger': st.session_state.get('clear_chat_trigger', False),
        'reset_flag': st.session_state.get('reset_flag', False),
        'is_processing': st.session_state.get('is_processing', False),
        'feedback_submitted': st.session_state.get('feedback_submitted', False),
        'last_response_index': st.session_state.get('last_response_index', -1),
        'timestamp': time.time()
    }

    try:
        with open(session_file, 'wb') as f:
            pickle.dump(state_to_save, f)
    except Exception as e:
        print(f"Error saving session: {e}")

def load_session_state():
    """Load session state data for specific user"""
    try:
        if not st.session_state.get('user_id'):
            return

        device_fingerprint = generate_device_fingerprint()
        session_file = get_session_file_path(st.session_state.user_id, device_fingerprint)
        if os.path.exists(session_file):
            with open(session_file, 'rb') as f:
                saved_session = pickle.load(f)
                if saved_session.get('user_id') == st.session_state.user_id and is_session_valid(saved_session):
                    for key, value in saved_session.items():
                        if key != 'timestamp':
                            st.session_state[key] = value
    except Exception as e:
        print(f"Error loading session: {e}")

def logout():
    """Safely logout user and clean up their session"""
    if st.session_state.get('user_id'):
        device_fingerprint = generate_device_fingerprint()
        session_file = get_session_file_path(st.session_state.user_id, device_fingerprint)
        if os.path.exists(session_file):
            try:
                os.remove(session_file)
            except Exception as e:
                print(f"Error removing session file: {e}")

    # Clear all session state variables
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def cleanup_old_sessions():
    """Cleanup expired session files"""
    try:
        sessions_dir = '.streamlit/sessions'
        if os.path.exists(sessions_dir):
            current_time = time.time()
            for filename in os.listdir(sessions_dir):
                filepath = os.path.join(sessions_dir, filename)
                if os.path.isfile(filepath):
                    try:
                        with open(filepath, 'rb') as f:
                            session_data = pickle.load(f)
                            if not is_session_valid(session_data):
                                os.remove(filepath)
                    except:
                        # If file is corrupted, remove it
                        os.remove(filepath)
    except Exception as e:
        print(f"Error cleaning up sessions: {e}")

@st.fragment
def clear_chat_history():
    st.session_state.messages = []
    st.session_state["chat"] = ""
    st.session_state.clear_chat_trigger = True
    st.session_state.reset_flag = True
    st.session_state.last_response_index = -1

    device_fingerprint = generate_device_fingerprint()
    session_data = {
        'authenticated': True,
        'user_id': st.session_state.user_id,
        'email': st.session_state.email,
        'device_fingerprint': device_fingerprint,
        'timestamp': time.time()
    }

    session_file = f'.streamlit/sessions/user_{st.session_state.user_id}_{device_fingerprint[:8]}.pkl'
    with open(session_file, 'wb') as f:
        pickle.dump(session_data, f)

    st.rerun()

@st.cache_resource
def get_rag_model():
    return load_model("rag")

@st.cache_resource
def get_encoder():
    return load_model("encoder")

@st.cache_resource
def get_llm():
    return load_model("osllm", model_name_or_path="meta-llama/Llama-3.2-3B-Instruct")

def handle_feedback(feedback):
    if feedback["score"] == "\U0001F44D":  # thumbs up unicode
        feedback["score"] = 1
    elif feedback["score"] == "\U0001F44E":  # thumbs down unicode
        feedback["score"] = 0

    if st.session_state.last_response_index >= 0:
        st.session_state.messages[st.session_state.last_response_index]["feedback"] = feedback
        st.session_state.feedback_submitted = False


def save_query(query):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("./logs/log.txt", "a", newline="") as outfile:
        outfile.write(f"{timestamp}: {query}")


def filter_data(categories, records, month_keys, value="Category", add_date=True):
    category, kv_pair = rag.init_counter(categories)

    data = [record for record in records if rag.match_record(categories, record, kv_pair, month_keys, value)]
    if add_date:
        data = ["\n".join([f"{key}: {val}"
                           for key, val in rag.add_date_format(rag.parse_record(datapoint)).items()])
                for datapoint in data]

    return category, kv_pair, data

def generate_records(main_query, dt_today):
    return main_query

def response_generator(main_query):
    return main_query

def stream_content(content, placeholder, speed=0.03):
    displayed_response = ""
    chunk_size = 3
    special_chars = {'*', '`', '_', '#', '<', '>', '|', '[', ']', '(', ')', '-'}

    lines = content.split('\n')
    for line_idx, line in enumerate(lines):
        if line.strip().startswith(('- ', '* ', '1.', '2.', '3.')):
            indent_level = len(line) - len(line.lstrip())
            displayed_response += ' ' * indent_level

        for i in range(0, len(line) + 1):  # +1 to include final character
            if i < len(line):
                displayed_response += line[i]

            if i % chunk_size == 0 or i == len(line):
                placeholder.markdown(displayed_response + "▌", unsafe_allow_html=True)
                time.sleep(speed)

        if line_idx < len(lines) - 1:
            displayed_response += '\n'

    placeholder.markdown(displayed_response, unsafe_allow_html=True)

@st.fragment
def clear_chat_btn():
    if st.button("Clear Chat"):
        clear_chat_history()

@st.fragment
def logout_btn():
    if st.button("Logout", type="primary"):
        logout()


def main():
    # Load saved state at the start
    load_session_state()

    # st.title("**Chat**")

    # Create main chat container
    chat_container = st.container()

    with chat_container:
        # Create a fixed height container for messages
        messages_container = st.container(height=600)

        # Get new user input - placing it below messages container
        user_query = st.chat_input("How can I help you today?")

        # button_cols = st.columns([6, 2, 2])  # First column for spacing
        # with button_cols[1]:
        #     clear_chat_btn()
        # with button_cols[2]:
        #     logout_btn()

        # Display all messages within the messages container
        with messages_container:
            if st.session_state.clear_chat_trigger:
                st.session_state.clear_chat_trigger = False
                st.session_state.reset_flag = False
                st.rerun()

            # If there's a new user query, add it and set processing flag
            if user_query:
                st.session_state.is_processing = True
                st.session_state.feedback_submitted = False
                st.session_state.messages.append({"role": "user", "content": user_query})
                save_session_state()

            # Display all messages
            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"], avatar=None):
                    st.markdown(message["content"])
                    if message.get("show_dashboard"):
                        create_dashboard(message.get("dataframe"), unique_id=f"dash_{i}")

                # Handle feedback for assistant messages
                if (message["role"] == "assistant" and
                        i == len(st.session_state.messages) - 1 and
                        not st.session_state.is_processing and
                        not st.session_state.feedback_submitted):
                    feedback = streamlit_feedback(feedback_type="thumbs", key=f"feedback_{i}")
                    if feedback:
                        st.session_state.feedback_submitted = True
                        handle_feedback(feedback)
                        save_session_state()

            if st.session_state.is_processing:
                assistant_placeholder = st.empty()

                with assistant_placeholder.container():
                    with st.chat_message("assistant", avatar=None):
                        dots_area = st.empty()

                query_text = st.session_state.messages[-1]["content"]

                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(process_query, query_text)

                    i = 0
                    word = random.choice(["Thinking", "Analyzing", "Working"])
                    while not future.done():
                        dots = '.' * ((i % 4) + 1)
                        dots_area.markdown(f"{word}{dots}")
                        i += 1
                        time.sleep(0.5)

                    show_dashboard = future.result()

                assistant_placeholder.empty()

                response = "Here's your Dashboard!\n\n# Analysis\n\n**Key metrics** are shown in the `dashboard`" if show_dashboard else "No dashboard required."
                dataframe = create_sample_data() if show_dashboard else None
                message = {"role": "assistant", "content": response, "show_dashboard": show_dashboard,
                           "dataframe": dataframe}

                with st.chat_message("assistant", avatar=None):
                    placeholder = st.empty()
                    stream_content(response, placeholder)

                st.session_state.messages.append(message)
                st.session_state.is_processing = False
                save_session_state()
                st.rerun()


if __name__ == "__main__":
    rag = get_rag_model()
    llm = get_llm()
    main()
