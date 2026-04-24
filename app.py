import streamlit as st
import os

st.set_page_config(page_title="Diagnostic Mode")
st.title("🟢 System Diagnostic Online")

st.write("If you are reading this on the public web, your Streamlit account, your GitHub bridge, and the web server are functioning perfectly.")

st.divider()

# Let's test if Streamlit is properly reading your hidden variables
if "GOOGLE_API_KEY" in st.secrets:
    st.success("Secure Connection Verified: Google API Key detected in Streamlit Secrets.")
else:
    st.error("Missing Credential: Google API Key NOT found in Secrets. Please check Advanced Settings on the deployment dashboard.")
