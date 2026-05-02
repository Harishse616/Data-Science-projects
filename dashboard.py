import altair as alt
import streamlit as st
import pandas as pd
import kagglehub
import folium
from streamlit_folium import st_folium
import json
import requests
import numpy as np

st.set_page_config(layout="wide")

# --- Top header / product style ---
st.markdown(
    """
    <style>
    .header-title {font-size:32px; font-weight:700; margin-bottom:6px}
    .header-sub {color:#6c757d; margin-top:0; margin-bottom:12px}
    .metric-card {background:#ffffff; padding:12px; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.08)}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container():
    st.markdown('<div class="header-title">COVID-19 Analytics — Global Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-sub">Interactive exploration of confirmed COVID-19 cases using Johns Hopkins data (via Kaggle). Use the sidebar to filter by time and countries.</div>', unsafe_allow_html=True)

# Download latest version (data logic unchanged)
path = kagglehub.dataset_download("antgoldbloom/covid19-data-from-john-hopkins-university")
st.markdown("---")
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv(f"{path}/CONVENIENT_global_confirmed_cases.csv")
    df = df.rename(columns={df.columns[0]: "Date"})
    df = df.iloc[1:].copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

@st.cache_data
def load_geo_data():
    # Load country coordinates for map
    geo_df = pd.read_csv(f"{path}/CONVENIENT_global_metadata.csv")
    return geo_df

data = load_data()
geo_data = load_geo_data()

# Sidebar controls - organized
st.sidebar.markdown("## Filters & Controls")

st.sidebar.markdown("---")
st.sidebar.subheader("Time Filter")
date_range = st.sidebar.slider(
    "Select date range:",
    min_value=data.index.min().to_pydatetime(),
    max_value=data.index.max().to_pydatetime(),
    value=(data.index.min().to_pydatetime(), data.index.max().to_pydatetime()),
    format="YYYY-MM-DD"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Country Selection")
all_countries = [col for col in data.columns if col not in ['Country/Region', 'Province/State']]
selected_countries = st.sidebar.multiselect(
    "Select countries to compare:",
    all_countries,
    default=["US", "China", "India", "Brazil", "France"]
)

if not selected_countries:
    selected_countries = all_countries[:10]

st.sidebar.markdown("---")
st.sidebar.markdown("#### About this dashboard\nInteractive product-style dashboard showing confirmed COVID-19 cases. Data source: Johns Hopkins (via Kaggle). Use filters above to explore")

# Filter data by date range
filtered_data = data[(data.index >= pd.Timestamp(date_range[0])) & (data.index <= pd.Timestamp(date_range[1]))]

# Get latest data
latest_date = filtered_data.index.max()
latest_row = filtered_data.loc[latest_date]
top_countries = latest_row.dropna().sort_values(ascending=False).head(10)
chart_df = top_countries.reset_index()
chart_df.columns = ["Country", "Confirmed cases"]

total_confirmed = int(latest_row.fillna(0).sum())
tracked_countries = int(data.shape[1])

# --- Trend Analysis (stacked vertically) ---
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
st.markdown("---")
st.header("Trend Analysis")
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# Top 10 bar chart (full width)
st.subheader("Top 10 Countries on Latest Date")
bar_chart = alt.Chart(chart_df).mark_bar().encode(
    x=alt.X("Confirmed cases:Q", title="Confirmed cases"),
    y=alt.Y("Country:N", sort="-x", title=None),
    tooltip=["Country:N", "Confirmed cases:Q"],
)
st.altair_chart(bar_chart, use_container_width=True)
st.caption("Bar chart showing the top 10 countries by confirmed cases on the selected date.")
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# Line chart for selected countries (stacked below)
st.subheader("Selected Countries Time Series")
available_selected = [c for c in selected_countries if c in data.columns]
if available_selected:
    series_data = filtered_data[available_selected]
    st.line_chart(series_data, use_container_width=True)
    st.caption("Line chart comparing trends for the selected countries over the chosen time range.")
else:
    st.info("No selected countries found in dataset")

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.subheader("Latest Data Summary")
st.dataframe(chart_df.head(15).set_index("Country"))
st.caption("Table summarizing the latest confirmed case counts for the top countries.")
st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    

# Map section
st.subheader("Geographic Map - Cases by Country (Time Period Selected)")
st.write("Red = Highest cases | Blue/Green = Lower cases | Based on selected date range")

# Get latest case counts for selected date range
latest_date_in_range = filtered_data.index.max()
latest_cases_in_range = filtered_data.loc[latest_date_in_range].dropna()

# Create a dataframe for the choropleth
map_data = pd.DataFrame({
    'Country': latest_cases_in_range.index,
    'Cases': latest_cases_in_range.values
}).reset_index(drop=True)


# Load GeoJSON data for world countries
try:
    geo_url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json'
    geo_data_geojson = requests.get(geo_url).json()
except:
    st.error("Unable to load geographic data. Please check your internet connection.")
    geo_data_geojson = None

if geo_data_geojson:
    # Country name mapping: dataset names -> GeoJSON names
    country_mapping = {}
    for feature in geo_data_geojson['features']:
        geojson_name = feature['properties']['name']
        country_mapping[geojson_name] = geojson_name
    
    # Define manual mapping for known mismatches
    dataset_to_geojson = {
        'US': 'United States of America',
        'China': 'China',
        'Russia': 'Russian Federation',
        'Korea, South': 'South Korea',
        'Korea, North': 'North Korea',
        'Taiwan*': 'Taiwan',
        'West Bank and Gaza': 'Palestine',
        'Congo (Kinshasa)': 'Dem. Rep. Congo',
        'Congo (Brazzaville)': 'Congo',
        'Czechia': 'Czech Republic',
        'Eswatini': 'Eswatini',
        'Burma': 'Myanmar',
        'Cote d\'Ivoire': 'Côte d\'Ivoire',
        'Reunion': 'Réunion',
        'Curacao': 'Curaçao',
        'Saint Barthelemy': 'St. Barthélemy',
        'Sint Maarten': 'Sint Maarten',
        'Timor-Leste': 'Timor-Leste',
        'Holy See': 'Vatican',
    }
    
    # Apply mapping to country names
    map_data['Country_Mapped'] = map_data['Country'].map(
        lambda x: dataset_to_geojson.get(x, x)
    )
    
    # Apply logarithmic scaling for better color distribution
    map_data['Cases_Log'] = np.log1p(map_data['Cases'])
    
    # Augment geojson features with case info when available (for tooltips)
    cases_lookup = dict(zip(map_data['Country_Mapped'], map_data['Cases']))
    for feature in geo_data_geojson['features']:
        name = feature['properties'].get('name')
        feature['properties']['cases'] = cases_lookup.get(name, None)

    # Create folium map
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="OpenStreetMap")

    # Create choropleth (using log-scaled values)
    folium.Choropleth(
        geo_data=geo_data_geojson,
        name='Cases',
        data=map_data,
        columns=['Country_Mapped', 'Cases_Log'],
        key_on='feature.properties.name',
        fill_color='RdYlBu_r',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Log Scale - Confirmed Cases',
        nan_fill_color='lightgray'
    ).add_to(m)

    # Add GeoJson layer with tooltip showing country name and raw cases
    tooltip = folium.features.GeoJsonTooltip(fields=['name','cases'], aliases=['Country','Cases'], localize=True)
    folium.GeoJson(
        geo_data_geojson,
        name='Country details',
        tooltip=tooltip,
        style_function=lambda feature: {
            'fillOpacity': 0.0, 'color': 'transparent'
        }
    ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    st.markdown("---")
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.header("Geographic Distribution")
    st.caption("Choropleth showing confirmed cases for the selected date range (log-scaled). Warmer colors indicate higher case counts.")
    st.write(f"**Data as of:** {latest_date_in_range.strftime('%Y-%m-%d')}")
    st.info(f"Map displays {len(map_data[map_data['Country_Mapped'].isin([f['properties']['name'] for f in geo_data_geojson['features']])])} countries with data")
    st_folium(m, width=1400, height=600)
else:
    st.info("Unable to display map due to data loading issues.")

# Footer / Credits
st.markdown("---")
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.markdown('<div style="text-align:right; color:#6c757d; font-size:12px;">Credits: Developed by Harish S .</div>', unsafe_allow_html=True)