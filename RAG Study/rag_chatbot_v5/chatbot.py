# app.py
import streamlit as st
import requests

# Configure the page
st.set_page_config(
    page_title="CV Chat", #title in the middle of the browser tab
    #add a version number 1.o,
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Backend configuration
BACKEND_URL = "http://localhost:5555"

def call_backend(endpoint, data=None, files=None):
    """Helper function to call backend API"""
    try:
        url = f"{BACKEND_URL}{endpoint}"
        if files:
            response = requests.post(url, files=files)
        else:
            response = requests.post(url, data=data)
        return response.json() if response.status_code == 200 else None
    except:
        return None

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "has_cv" not in st.session_state:
    st.session_state.has_cv = False
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "selected_file" not in st.session_state:
    st.session_state.selected_file = None

# Title
st.title("CV Chat")

# Chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Upload CV section with manual upload button
# st.subheader("Upload CV")

# File selection 
# file selection has been validated in the backend, it can upload all files but only process pdfs(please see main.py    )   
uploaded_file = st.file_uploader(
    "Choose a PDF file", 
    type="pdf", 
    label_visibility="collapsed"
)

if uploaded_file:
    st.session_state.selected_file = uploaded_file
    st.info(f"Selected: {uploaded_file.name}")

# Manual upload button
if st.session_state.selected_file and not st.session_state.has_cv:
    if st.button("Upload PDF", type="primary", use_container_width=True):
        with st.status("Uploading and processing CV...") as status:
            files = {"file": (st.session_state.selected_file.name, st.session_state.selected_file.getvalue(), "application/pdf")}
            result = call_backend("/upload-pdf", files=files)
            
            if result and "message" in result:
                st.session_state.has_cv = True
                st.session_state.uploaded_file_name = st.session_state.selected_file.name
                status.update(label=" PDF has been Uploaded", state="complete")
                st.success(result["message"])
            else:
                status.update(label=" Upload Failed . Please try again", state="error")
                st.error("Failed to upload CV")

# Status indicator
if st.session_state.has_cv:
    st.success(f" **{st.session_state.uploaded_file_name}** loaded")
else:
    st.info("Select a CV PDF and click Upload to get started")

st.divider()

# Question input (only show if CV is loaded)
if st.session_state.has_cv:
    question = st.text_input(
        "Ask about the uploaded CV...",
        label_visibility="collapsed",
        placeholder="Ask about the uploaded CV..."
    )

    # Action buttons
    col1, col2 = st.columns([1, 4])

    with col1:
        if st.button(" Reset", use_container_width=True, type="secondary"):
            result = call_backend("/reset")
            if result and "message" in result:
                st.session_state.messages = []
                st.session_state.has_cv = False
                st.session_state.uploaded_file_name = None
                st.session_state.selected_file = None
                st.success(" All data reset")
                st.rerun()

    with col2:
        send_disabled = not st.session_state.has_cv or not question.strip()
        if st.button("Ask", 
                    use_container_width=True, 
                    type="primary",
                    disabled=send_disabled):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("💭 Thinking..."):
                    result = call_backend("/ask", data={"question": question})
                    if result and "answer" in result:
                        answer = result["answer"]
                        st.write(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        error_msg = " Failed to get response from AI"
                        st.write(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            st.rerun()
else:
    st.info("💡 Upload a CV first to start asking questions") #Modify it to a little description of the system