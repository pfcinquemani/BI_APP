import codecs
import streamlit as st
import streamlit.components.v1 as stc

st.title("Please choose an option from the sidebar")

menu = ["Choose an option", "Finance Analysis Dashboard", "Dashboard with Geolocalization", "All"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Finance Analysis Dashboard":
    st.subheader("Finance Analysis Dashboard")

    pagina = "power_bi1.html"
    file = codecs.open(pagina, 'r')
    page = file.read()
    stc.html(page, width=800, height=500, scrolling=False)

if choice == "Dashboard with Geolocalization":
    st.subheader("Dashboard with Geolocalization")
    pagina = "power_bi2.html"
    file = codecs.open(pagina, 'r')
    page = file.read()
    stc.html(page, width=800, height=500, scrolling=False)



if choice == "All":
    st.subheader("Finance Analysis Dashboard")
    pagina = "power_bi1.html"
    file = codecs.open(pagina, 'r')
    page = file.read()
    stc.html(page, width=800, height=500, scrolling=False)

    st.subheader("Dashboard with Geolocalization")
    pagina = "power_bi2.html"
    file = codecs.open(pagina, 'r')
    page = file.read()
    stc.html(page, width=800, height=500, scrolling=False)
