import streamlit as st
import pandas as pd
import plotly.express as px

st.title("IPL Cricket Data Analysis Dashboard")

@st.cache_data
def load_data():
    matches = pd.read_csv("matches.csv")
    return matches

matches = load_data()

# Team Wins
team_wins = matches['winner'].value_counts().reset_index()
team_wins.columns = ['Team', 'Wins']

fig1 = px.bar(
    team_wins,
    x='Team',
    y='Wins',
    title='IPL Team Wins'
)

st.plotly_chart(fig1)

# Toss Analysis
toss = matches['toss_winner'].value_counts().reset_index()
toss.columns = ['Team', 'Toss Wins']

fig2 = px.pie(
    toss,
    names='Team',
    values='Toss Wins',
    title='Toss Winner Distribution'
)

st.plotly_chart(fig2)

# Venue Analysis
venue = matches['venue'].value_counts().reset_index()
venue.columns = ['Venue', 'Matches']

fig3 = px.bar(
    venue,
    x='Venue',
    y='Matches',
    title='Matches by Venue'
)

st.plotly_chart(fig3)
