import streamlit as st
import pickle

booking = load_booking()

# Load the trained model
clf = pickle.load(open('classifier.pkl', 'wb'))

# Sidebar for user input
st.sidebar.title('Iris Classifier')
lead_time = st.sidebar.slider('lead_time', 0.1, 250, 20)
market_segment_type = st.sidebar.slider('market_segment_type', 1.0, 4.0, 3.0)
special_requests = st.sidebar.slider('special_requests', 0.1, 3.0, 1.0)
avg_price_per_room = st.sidebar.slider('avg_price_per_room', 100, 4000, 500)
arrival_month = st.sidebar.slider('arrival_month', 1.0, 12, 6)

# Make predictions
prediction = clf.predict([[lead_time, market_segment_type, special_requests, avg_price_per_room, arrival_month]])

# Display prediction
st.write('## Prediction:')
st.write(booking.target_status[prediction[0]])
