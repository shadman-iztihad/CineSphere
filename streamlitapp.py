import os
import sys
import logging
import streamlit as st

# Add relevant folders to sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
components_path = os.path.join(project_root, "components")
recommendation_path = os.path.join(project_root, "recommendation")

sys.path.insert(0, components_path)
sys.path.insert(0, recommendation_path)

# Import components and utilities
from components.auth_ui import AuthUI
from components.sidebar_menu import (
    welcome_menu, 
    profile_section, 
    recommendation_tab, 
    performance_analytics_tab
)

auth_ui = AuthUI()

# Initialize session state variables
if "LOGGED_IN" not in st.session_state:
    st.session_state["LOGGED_IN"] = False
if "USERNAME" not in st.session_state:
    st.session_state["USERNAME"] = "Guest"
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True  # Default to dark mode
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Login"

# Apply Theme
def apply_theme():
    if st.session_state["dark_mode"]:
        st.markdown(
            """<style>
            body { background-color: #1E1E2E; color: #FFFFFF; font-family: 'Arial', sans-serif; }
            .stSidebar { background-color: #2B2B3D; color: #FFFFFF; border-right: 2px solid #444; }
            .stButton > button { background-color: #4CAF50; color: white; padding: 10px; }
            .stButton > button:hover { background-color: #45A049; }
            .stSelectbox, .stMultiselect { background-color: #2B2B3D; color: #FFFFFF; border: 1px solid #555; }
            </style>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<style>
            body { background-color: #FFFFFF; color: #000000; font-family: 'Arial', sans-serif; }
            .stSidebar { background-color: #F1F1F1; color: #000000; border-right: 1px solid #DDD; }
            .stButton > button { background-color: #008CBA; color: white; padding: 10px; }
            .stButton > button:hover { background-color: #005F6A; }
            .stSelectbox, .stMultiselect { background-color: #FFFFFF; color: #000000; border: 1px solid #DDD; }
            </style>""",
            unsafe_allow_html=True,
        )

# Sidebar Navigation
def sidebar_navigation():
    if not st.session_state.get("LOGGED_IN", False):
        st.sidebar.title("Menu")
        auth_option = st.sidebar.radio("Choose an option:", ["Login", "Sign Up", "Forgot Password"], key="auth_menu")
        if auth_option == "Login":
            auth_ui.login_widget()
        elif auth_option == "Sign Up":
            auth_ui.sign_up_widget()
        elif auth_option == "Forgot Password":
            auth_ui.forgot_password_widget()
    else:
        username = st.session_state["USERNAME"]
        st.sidebar.title(f"Welcome, {username}")

        menu_option = st.sidebar.radio(
            "Navigate to:",
            ["Welcome", "My Profile", "Recommendations", "Performance & Analytics", "Logout"],
            key="main_menu",
        )

        # Update active tab based on menu selection
        if menu_option == "Welcome":
            st.session_state["active_tab"] = "Welcome"
        elif menu_option == "My Profile":
            st.session_state["active_tab"] = "My Profile"
        elif menu_option == "Recommendations":
            st.session_state["active_tab"] = "Recommendations"
        elif menu_option == "Performance & Analytics":
            st.session_state["active_tab"] = "Performance & Analytics"
        elif menu_option == "Logout":
            st.session_state["LOGGED_IN"] = False
            st.session_state["USERNAME"] = "Guest"
            st.session_state["active_tab"] = "Login"
            st.session_state["dark_mode"] = True  # Reset theme
            st.experimental_rerun()

# Render Active Tab
def render_active_tab():
    try:
        if st.session_state.get("active_tab") == "Welcome":
            welcome_menu()
        elif st.session_state.get("active_tab") == "My Profile":
            profile_section()
        elif st.session_state.get("active_tab") == "Recommendations":
            recommendation_tab()
        elif st.session_state.get("active_tab") == "Performance & Analytics":
            performance_analytics_tab()
        else:
            st.warning("You need to log in to access this page.")
    except Exception as e:
        st.error("An unexpected error occurred while loading the selected tab.")
        logging.error(f"Error in active tab {st.session_state.get('active_tab', 'Unknown')}: {e}")

# Main Application Logic
apply_theme()  # Apply theme globally
sidebar_navigation()
if st.session_state.get("LOGGED_IN", False):
    render_active_tab()