# traffic_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import numpy as np
from datetime import datetime
import osmnx as ox
import networkx as nx
import time
import warnings
warnings.filterwarnings('ignore')

# === Page Configuration ===
st.set_page_config(
    page_title="India InterCity Traffic Dashboard", 
    layout="wide",
    page_icon="🚗"
)

# === Custom CSS for better styling ===
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .prediction-high {
        background-color: #ff6b6b;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .prediction-medium {
        background-color: #ffa726;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .prediction-low {
        background-color: #66bb6a;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# === Load Data and Model with Error Handling ===
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r"C:\Users\shing\OneDrive\Documents\Traffic Prediction\traffic_dashboard\app\indian_traffic_data.csv")
        df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
        df = df.dropna(subset=['date_time'])
        df['month'] = df['date_time'].dt.month
        df['hour'] = df['date_time'].dt.hour
        df['day_name'] = df['date_time'].dt.day_name()
        # Normalize weather columns to lowercase to handle inconsistencies
        df['weather_main'] = df['weather_main'].str.lower()
        df['weather_description'] = df['weather_description'].str.lower()

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_model():
    try:
        model_data = joblib.load(r"C:\Users\shing\OneDrive\Documents\Traffic Prediction\traffic_dashboard\app\traffic_volume_model.pkl")
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load data and model
df = load_data()
model_data = load_model()

if df is None or model_data is None:
    st.stop()

model = model_data['model']
encoders = model_data.get('encoders', {})
feature_names = model_data.get('feature_names', [])

# === Routing Function ===
@st.cache_data(show_spinner=False)
def get_road_route(start_coords, end_coords, max_single_graph_m=80000):
    """
    Robust inter-city routing using OSMnx and NetworkX.
      - build two local graphs (around start and end) with adaptive radius
      - compose them and attempt shortest_path on the composed graph
      - fall back to straight line (return None) with informative messages
    Returns: list of (lat, lon) tuples OR None (meaning use straight line)
    """
    try:
        # helper: haversine distance in km
        from math import radians, sin, cos, sqrt, atan2
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0
            dlat = radians(lat2 - lat1)
            dlon = radians(lon2 - lon1)
            a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            return R * c

        dist_km = haversine(start_coords['lat'], start_coords['lon'], end_coords['lat'], end_coords['lon'])

        # If extremely far ( > ~1500 km ) prefer external routing service (OSRM/GraphHopper).
        if dist_km > 1500:
            st.info(f"Route is ~{dist_km:.0f} km — consider using a dedicated routing API (OSRM/GraphHopper) for full road routing.")
            return None

        # Build two smaller graphs using graph_from_point to avoid massive bbox downloads
        # radius chosen adaptively: at least 5km up to max_single_graph_m (meters)
        start_radius = min(max(5000, int((dist_km * 1000) * 0.25)), max_single_graph_m)
        end_radius = start_radius

        G1 = ox.graph_from_point((start_coords['lat'], start_coords['lon']), dist=start_radius, network_type='drive', simplify=True)
        G2 = ox.graph_from_point((end_coords['lat'], end_coords['lon']), dist=end_radius, network_type='drive', simplify=True)

        # If graphs are tiny (rural), increase radius slightly and retry
        if len(G1) < 5 or len(G2) < 5:
            start_radius = min(max_single_graph_m, start_radius * 2)
            end_radius = start_radius
            G1 = ox.graph_from_point((start_coords['lat'], start_coords['lon']), dist=start_radius, network_type='drive', simplify=True)
            G2 = ox.graph_from_point((end_coords['lat'], end_coords['lon']), dist=end_radius, network_type='drive', simplify=True)

        # Compose (merge) graphs into one connected graph
        G = nx.compose(G1, G2)

        # Nearest node lookup: handle different osmnx versions and signatures
        try:
            start_node = ox.distance.nearest_nodes(G, start_coords['lon'], start_coords['lat'])
            end_node = ox.distance.nearest_nodes(G, end_coords['lon'], end_coords['lat'])
        except Exception:
            try:
                start_node = ox.nearest_nodes(G, X=start_coords['lon'], Y=start_coords['lat'])
                end_node = ox.nearest_nodes(G, X=end_coords['lon'], Y=end_coords['lat'])
            except Exception:
                # final fallback: nearest by iterating nodes (slow)
                def brute_nearest(Gx, lat, lon):
                    best = None
                    best_d = float('inf')
                    for n, d in Gx.nodes(data=True):
                        dy = (d.get('y', 0) - lat)
                        dx = (d.get('x', 0) - lon)
                        dd = dx*dx + dy*dy
                        if dd < best_d:
                            best_d = dd
                            best = n
                    return best
                start_node = brute_nearest(G, start_coords['lat'], start_coords['lon'])
                end_node = brute_nearest(G, end_coords['lat'], end_coords['lon'])

        # Shortest path (by length)
        try:
            route = nx.shortest_path(G, source=start_node, target=end_node, weight='length')
            route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
            # If the route returned only start/end and they are far apart, consider fallback
            if len(route_coords) < 3 and dist_km > 50:
                st.warning("OSMnx route too sparse for long-distance route — falling back to direct line.")
                return None
            return route_coords
        except nx.NetworkXNoPath:
            st.warning("No path in composed graph. Falling back to straight line.")
            return None
        except Exception as e:
            st.warning(f"Routing failed: {e}. Falling back to straight line.")
            return None

    except Exception as e:
        st.warning(f"Could not calculate route due to: {e}. Showing a straight line instead.")
        return None


# ------------------------------
# Realtime simulation helper
# ------------------------------
def simulate_realtime_along_route(route_coords, iterations=200, delay=0.5):
    """
    Simulate a moving vehicle along route_coords.
    - route_coords: list of (lat, lon)
    - iterations: how many map updates to run (or stop earlier)
    - delay: seconds between frames
    This is intended for interactive Streamlit runs (local or server).
    """
    if not route_coords or len(route_coords) < 2:
        st.info("No detailed route available to simulate.")
        return

    placeholder = st.empty()
    # interpolate points to get smooth movement
    lats = [c[0] for c in route_coords]
    lons = [c[1] for c in route_coords]

    # create a long list of points by linear interpolation between nodes
    interp_lats = []
    interp_lons = []
    steps_between = max(3, int(len(route_coords)))  # tweak to smoothness
    for i in range(len(lats) - 1):
        for t in range(steps_between):
            frac = t / float(steps_between)
            interp_lats.append(lats[i] * (1 - frac) + lats[i+1] * frac)
            interp_lons.append(lons[i] * (1 - frac) + lons[i+1] * frac)
    # append final point
    interp_lats.append(lats[-1])
    interp_lons.append(lons[-1])

    total_points = len(interp_lats)
    idx = 0
    for frame in range(iterations):
        i = idx % total_points
        cur_lat = interp_lats[i]
        cur_lon = interp_lons[i]

        fig = go.Figure()
        # route line
        fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon=lons, lat=lats,
            line=dict(width=3),
            name="Route"
        ))
        # moving marker
        fig.add_trace(go.Scattermapbox(
            mode="markers+text",
            lon=[cur_lon],
            lat=[cur_lat],
            marker=dict(size=14),
            text=["Vehicle"],
            textposition="top center",
            name="Vehicle"
        ))

        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_center_lon=cur_lon, 
            mapbox_center_lat=cur_lat,
            mapbox_zoom=8,
            margin={"r":0,"t":0,"l":0,"b":0}
        )
        placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(delay)
        idx += 1

# === Header ===
st.markdown('<div class="main-header">🚗 Real-Time InterCity Traffic Prediction & Analysis for Indian Cities</div>', unsafe_allow_html=True)

# === Sidebar Filters ===
st.sidebar.header("🔍 Filter Data")

# City selection with validation
cities = sorted(df['from_city'].unique())
from_city = st.sidebar.selectbox("From City", cities, index=0)
to_city = st.sidebar.selectbox("To City", [c for c in cities if c != from_city], index=1)

# Month selection with names
month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
month = st.sidebar.selectbox("Month", month_names, index=0)
month_num = month_names.index(month) + 1

# Weather filter
weather_options = ['All'] + sorted(df['weather_main'].unique().tolist())
selected_weather = st.sidebar.multiselect("Weather Conditions", weather_options, default=['All'])

# Date range filter
min_date = df['date_time'].min().date()
max_date = df['date_time'].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date])

# === Filter Data ===
filtered = df[(df['from_city'] == from_city) & (df['to_city'] == to_city)]
if 'All' not in selected_weather:
    filtered = filtered[filtered['weather_main'].isin(selected_weather)]

if len(date_range) == 2:
    filtered = filtered[
        (filtered['date_time'].dt.date >= date_range[0]) & 
        (filtered['date_time'].dt.date <= date_range[1])
    ]

filtered_month = filtered[filtered['month'] == month_num]

# === Key Metrics ===
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Route Statistics")
if not filtered.empty:
    avg_traffic = filtered['traffic_volume'].mean()
    max_traffic = filtered['traffic_volume'].max()
    avg_aqi = filtered['aqi'].mean()
    
    st.sidebar.metric("Average Traffic", f"{avg_traffic:,.0f}")
    st.sidebar.metric("Peak Traffic", f"{max_traffic:,.0f}")
    st.sidebar.metric("Average AQI", f"{avg_aqi:.0f}")

# === Main Dashboard Layout ===
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.subheader(f"📊 Traffic Trend: {from_city} ➡️ {to_city} ({month})")
    if not filtered_month.empty:
        # Aggregate by day for cleaner plot
        daily_traffic = filtered_month.groupby(filtered_month['date_time'].dt.date).agg({
            'traffic_volume': 'mean',
            'weather_main': 'first',
            'aqi': 'mean'
        }).reset_index()
        
        fig1 = px.line(daily_traffic, x='date_time', y='traffic_volume', 
                      color='weather_main' if len(selected_weather) > 1 else None,
                      title=f"Daily Traffic Volume - {month}",
                      labels={'traffic_volume': 'Traffic Volume', 'date_time': 'Date'},
                      color_discrete_sequence=px.colors.qualitative.Set1)
        fig1.update_layout(hovermode='x unified')
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.warning("No data available for the selected filters")

with col2:
    st.subheader("🌡️ Weather Distribution")
    if not filtered_month.empty:
        weather_counts = filtered_month['weather_main'].value_counts()
        fig_pie = px.pie(values=weather_counts.values, names=weather_counts.index,
                        color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie, use_container_width=True)

with col3:
    st.subheader("🕒 Hourly Pattern")
    if not filtered_month.empty:
        hourly_avg = filtered_month.groupby('hour')['traffic_volume'].mean().reset_index()
        fig_bar = px.bar(hourly_avg, x='hour', y='traffic_volume',
                        title="Average Traffic by Hour",
                        color_discrete_sequence=['#2E86AB'])
        st.plotly_chart(fig_bar, use_container_width=True)

# === AQI vs Traffic Analysis ===
st.subheader("🌫️ AQI vs Traffic Volume Analysis")
if not filtered_month.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        fig2 = px.scatter(filtered_month, x='aqi', y='traffic_volume', 
                         color='weather_main', size='temp',
                         title="AQI Impact on Traffic Volume",
                         hover_data=['hour', 'temp'],
                         trendline="lowess",
                         color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        # Correlation heatmap for numerical features
        numerical_features = filtered_month[['traffic_volume', 'aqi', 'temp', 'rain_1h', 'clouds_all']].corr()
        fig_heatmap = px.imshow(numerical_features, 
                               text_auto=True, 
                               aspect="auto",
                               title="Feature Correlation Heatmap",
                               color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_heatmap, use_container_width=True)

# === Prediction Section ===
st.subheader("🔮 Traffic Volume Prediction")
st.markdown("Adjust the parameters below to predict traffic volume:")

pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)

with pred_col1:
    temp = st.slider("Temperature (°C)", 0, 45, 25, help="Average temperature")
    rain = st.slider("Rain (mm/hr)", 0.0, 20.0, 0.0, help="Rainfall intensity")

with pred_col2:
    clouds = st.slider("Cloud Cover (%)", 0, 100, 50)
    aqi = st.slider("AQI", 0, 500, 100, help="Air Quality Index")

with pred_col3:
    hour = st.slider("Hour of Day", 0, 23, 8)
    day_of_week = st.selectbox("Day of Week", 
                              ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                              index=1)

with pred_col4:
    weather_main = st.selectbox("Weather Condition", sorted(df['weather_main'].unique()))
    is_holiday = st.checkbox("Holiday")

# Prepare prediction sample
day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}

sample_data = {
    'holiday': 'Holiday' if is_holiday else 'None',
    'temp': temp,
    'rain_1h': rain,
    'snow_1h': 0,
    'clouds_all': clouds,
    'weather_main': weather_main.lower(),
    'weather_description': weather_main,  # Simplified for demo
    'hour': hour,
    'day_of_week': day_mapping[day_of_week],
    'month': month_num,
    'is_rush_hour': 1 if hour in [7, 8, 9, 17, 18, 19] else 0,
    'aqi': aqi,
    'from_city': from_city,
    'to_city': to_city,
    'is_weekend': 1 if day_of_week in ['Saturday', 'Sunday'] else 0
}

sample_df = pd.DataFrame([sample_data])

# Encode categorical variables
try:
    for col in ['holiday', 'weather_main', 'weather_description', 'from_city', 'to_city']:
        if col in encoders:
            encoder = encoders[col]
            sample_df[col] = sample_df[col].apply(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
            )
        else:
            sample_df[col] = sample_df[col].astype('category').cat.codes
except Exception as e:
    st.error(f"Encoding error: {e}")


# Make prediction
if st.button("Predict Traffic Volume", type="primary"):
    try:
        # Ensure we have the right features in the right order
        if feature_names:
            sample_df = sample_df.reindex(columns=feature_names, fill_value=0)
        
        prediction = model.predict(sample_df)[0]
        
        # Display prediction with styling
        st.markdown("### Prediction Result")
        
        if prediction > 2000:
            st.markdown(f'<div class="prediction-high">🚨 High Traffic: {prediction:,.0f} vehicles/hour</div>', unsafe_allow_html=True)
        elif prediction > 1000:
            st.markdown(f'<div class="prediction-medium">⚠️ Moderate Traffic: {prediction:,.0f} vehicles/hour</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="prediction-low">✅ Light Traffic: {prediction:,.0f} vehicles/hour</div>', unsafe_allow_html=True)
            
        # Add context
        if not filtered.empty:
            avg_actual = filtered['traffic_volume'].mean()
            diff = prediction - avg_actual
            st.metric("Compared to Route Average", f"{prediction:,.0f}", 
                     delta=f"{diff:+.0f} vs average")
    
    except Exception as e:
        st.error(f"Prediction error: {e}")

# === Map Visualization ===
st.subheader("🗺️ InterCity Traffic Network")

col1, col2 = st.columns(2)
with col2:
    # City traffic summary
    st.markdown("**City Traffic Overview**")
    city_traffic = df.groupby('from_city').agg({
        'traffic_volume': 'mean',
        'aqi': 'mean',
        'to_city': 'count'
    }).nlargest(10, 'traffic_volume').reset_index()
    
    fig_city = px.bar(city_traffic, x='from_city', y='traffic_volume',
                     color='aqi',
                     title="Top 10 Cities by Average Traffic",
                     labels={'traffic_volume': 'Avg Traffic Volume', 'from_city': 'City'},
                     color_continuous_scale='Viridis')
    st.plotly_chart(fig_city, use_container_width=True)

with col1:
    st.markdown("**Route Visualization on Map**")

    # City coordinates for map plotting
    city_coords = {
        'Mumbai': {'lat': 19.0760, 'lon': 72.8777},
        'Delhi': {'lat': 28.7041, 'lon': 77.1025},
        'Bangalore': {'lat': 12.9716, 'lon': 77.5946},
        'Kolkata': {'lat': 22.5726, 'lon': 88.3639},
        'Chennai': {'lat': 13.0827, 'lon': 80.2707},
        'Hyderabad': {'lat': 17.3850, 'lon': 78.4867},
        'Pune': {'lat': 18.5204, 'lon': 73.8567},
        'Ahmedabad': {'lat': 23.0225, 'lon': 72.5714},
        'Jaipur': {'lat': 26.9124, 'lon': 75.7873},
        'Lucknow': {'lat': 26.8467, 'lon': 80.9462}
    }

    # Get coordinates for the selected route
    from_coords = city_coords.get(from_city)
    to_coords = city_coords.get(to_city)
    
    fig_map = go.Figure()

    if from_coords and to_coords:
        # Try to get the detailed road route
        route_coords = get_road_route(from_coords, to_coords)
        
        if route_coords:
            # Extract lat and lon from route coordinates
            lats = [coord[0] for coord in route_coords]
            lons = [coord[1] for coord in route_coords]
            
            # Add the detailed road route as a blue line
            fig_map.add_trace(go.Scattermapbox(
                mode="lines",
                lon=lons,
                lat=lats,
                line=dict(width=3, color='blue'),
                name=f"{from_city} to {to_city} (Road Route)",
                hovertemplate="Road Route<br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>"
            ))
        else:
            # Fallback to straight line if routing fails
            fig_map.add_trace(go.Scattermapbox(
                mode="lines",
                lon=[from_coords['lon'], to_coords['lon']],
                lat=[from_coords['lat'], to_coords['lat']],
                line=dict(width=3, color='green'),
                name=f"{from_city} to {to_city} (Direct)",
                hovertemplate="Direct Route<br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>"
            ))
        
        # Add start and end city markers
        fig_map.add_trace(go.Scattermapbox(
            mode="markers+text",
            lon=[from_coords['lon'], to_coords['lon']],
            lat=[from_coords['lat'], to_coords['lat']],
            marker=dict(size=12, color=['green', 'red']),
            text=[from_city, to_city],
            textposition="top center",
            name="Cities",
            hovertemplate="%{text}<br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>"
        ))

        # Simulation control: allow user to animate a vehicle along the OSMnx route
        if route_coords:
            if st.button("Simulate Live Vehicle"):
                # Run a short simulation; runs in the current Streamlit session
                simulate_realtime_along_route(route_coords, iterations=300, delay=0.3)

    fig_map.update_layout(
        title=f"Route Map: {from_city} to {to_city}",
        mapbox_style="open-street-map",
        mapbox_center_lon=78.9629, # Center of India
        mapbox_center_lat=20.5937,
        mapbox_zoom=3.5,
        margin={"r":0,"t":40,"l":0,"b":0},
        showlegend=True,
        hovermode='closest'
    )
    st.plotly_chart(fig_map, use_container_width=True)

# === Data Summary ===
with st.expander("📋 Data Summary"):
    st.write(f"**Dataset Overview:** {len(df):,} total records")
    st.write(f"**Filtered Data:** {len(filtered):,} records matching current filters")
    st.write(f"**Cities Coverage:** {len(cities)} cities")
    st.write(f"**Date Range:** {df['date_time'].min().strftime('%Y-%m-%d')} to {df['date_time'].max().strftime('%Y-%m-%d')}")

st.markdown("---")
st.caption("🚦 Indian InterCity Traffic Analysis Dashboard | Built with Streamlit & Plotly")