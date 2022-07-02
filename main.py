import codecs
import streamlit as st
import streamlit.components.v1 as stc

st.title("Please choose an option from the sidebar")

menu = ["Choose an option", "Dashboard 1", "Dashboard 2", "Dashboard 1 and 2"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Dashboard 1":
    st.subheader("Dashboard number 1")

    pagina = "power_bi1.html"
    file = codecs.open(pagina, 'r')
    page = file.read()
    stc.html(page, width=800, height=500, scrolling=False)

if choice == "Dashboard 2":
    st.subheader("Dashboard number 1")
    pagina = "power_bi2.html"
    file = codecs.open(pagina, 'r')
    page = file.read()
    stc.html(page, width=800, height=500, scrolling=False)

if choice == "Dashboard 1 and 2":
    st.subheader("Dashboard 2")
    pagina = "power_bi1.html"
    file = codecs.open(pagina, 'r')
    page = file.read()
    stc.html(page, width=800, height=500, scrolling=False)

    st.subheader("Dashboard 2")
    pagina = "power_bi2.html"
    file = codecs.open(pagina, 'r')
    page = file.read()
    stc.html(page, width=800, height=500, scrolling=False)


