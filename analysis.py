import streamlit as st
import pandas as pd

# Set page configuration to center content
st.set_page_config(layout="centered")

# Add custom CSS to center the input box
st.markdown("""
    <style>
    .css-1n76uvr {
        width: 60%;
        margin: auto;
    }

    /* Center title */
    .css-10trblm {
        text-align: center;
        margin-left: auto;
        margin-right: auto;
        width: 60% !important;
    }
    </style>
""", unsafe_allow_html=True)


# Function to generate DataFrame based on input
def generate_dataframe(input_text):
    # This is a sample function - modify according to your needs
    # Here we're just creating a simple example DataFrame
    words = input_text.split()
    data = {
        'Word': words,
        'Length': [len(word) for word in words],
        'Upper': [word.upper() for word in words]
    }
    return pd.DataFrame(data).reset_index(drop=True)  # Reset and drop index


# Title
st.title("Text Analysis App")

# Centered container for input
with st.container():
    # Add some spacing
    st.write("")
    st.write("")

    # Create the text input
    user_input = st.text_input(
        "Enter your text here:",
        key="text_input",
        placeholder="Type something..."
    )

    # Add some spacing after input
    st.write("")
    st.write("")

# Process input and display DataFrame
if user_input:
    # Generate DataFrame
    df = generate_dataframe(user_input)

    # Display DataFrame without index
    st.write("Analysis Results:")
    st.dataframe(df.set_index('Word').reset_index(), hide_index=True, use_container_width=True)
else:
    st.info("Please enter a query.")

# Add footer spacing
st.write("")
st.write("")
