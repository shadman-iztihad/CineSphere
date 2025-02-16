import streamlit as st
from utils import validate_user, register_user, reset_password

class AuthUI:
    def __init__(self):
        pass

    def login_widget(self):
        st.title("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="login_button"):
            if validate_user(username, password):
                st.session_state["LOGGED_IN"] = True
                st.session_state["USERNAME"] = username
                st.session_state["active_tab"] = "Welcome"  # Redirect to Welcome tab
                st.success("Login successful!")
                st.experimental_rerun()  # Refresh the app to reflect the logged-in state
            else:
                st.error("Invalid username or password.")

    def sign_up_widget(self):
        st.title("Sign Up")
        name = st.text_input("Full Name", key="signup_name")
        email = st.text_input("Email", key="signup_email")
        username = st.text_input("Username", key="signup_username")
        password = st.text_input("Password", type="password", key="signup_password")
        if st.button("Sign Up", key="signup_button"):
            if register_user(name, email, username, password):
                st.success("Account created successfully! Please log in.")
            else:
                st.error("Failed to create account. Please try again.")

    def forgot_password_widget(self):
        st.title("Forgot Password")
        email = st.text_input("Enter your registered email", key="forgot_email")
        new_password = st.text_input("New Password", type="password", key="forgot_new_password")
        if st.button("Reset Password", key="forgot_reset_button"):
            if reset_password(email, new_password):
                st.success("Password reset successfully! Please log in.")
            else:
                st.error("Failed to reset password. Please try again.")

    def logout_widget(self):
        if st.session_state.get("LOGGED_IN", False):
            if st.button("Logout", key="logout_button"):
                st.session_state["LOGGED_IN"] = False
                st.session_state["USERNAME"] = "Guest"
                st.session_state["active_tab"] = "Login"
                st.success("Logged out successfully!")
                st.experimental_rerun()