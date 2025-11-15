"""Watershed Water Quality Analysis Dashboard
==========================================
A comprehensive Streamlit application for analyzing and visualizing water quality data
across watersheds, including hotspot identification and conservation priorities.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Mapping
import folium
from streamlit_folium import st_folium
from folium import plugins

# Statistics
from scipy import stats

# Configure page
st.set_page_config(
    page_title="Watershed Quality Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {padding-top: 0;}
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Configure visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# Title and description
st.title("Watershed Water Quality Analysis Dashboard")
st.markdown("**Comprehensive analysis of water quality parameters, hotspots, and conservation priorities**")

# Sidebar for data upload and filters
with st.sidebar:
    st.header("Data Configuration")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Water Quality Data (CSV)",
        type=['csv'],
        help="Upload your water quality dataset in CSV format"
    )
    
    # Use sample data option
    use_sample = st.checkbox("Use Sample Data", value=True if not uploaded_file else False)
    
    if uploaded_file is not None or use_sample:
        st.success("Data loaded successfully!")
        
        st.markdown("---")
        st.header("Analysis Settings")
        
        # Thresholds
        st.subheader("EPA Thresholds")
        do_threshold = st.slider("DO Minimum (mg/L)", 0.0, 10.0, 5.0, 0.5)
        ph_min = st.slider("pH Minimum", 5.0, 7.0, 6.5, 0.1)
        ph_max = st.slider("pH Maximum", 7.5, 10.0, 8.5, 0.1)
        conductivity_max = st.slider("Conductivity Max (S/cm)", 200, 1000, 500, 50)
        tds_max = st.slider("TDS Max (mg/L)", 200, 1000, 500, 50)

@st.cache_data
def load_data(file=None):
    """Load and preprocess the water quality data."""
    if file is not None:
        df = pd.read_csv(file)
    else:
        # Default to sample_water_quality.csv if available
        import os
        default_file = 'sample_water_quality.csv'
        if os.path.exists(default_file):
            df = pd.read_csv(default_file)
        else:
            # Fallback: Generate sample data if file not found
            np.random.seed(42)
            n_samples = 5000
            n_stations = 50
            n_watersheds = 10
            
            # Generate sample data
            df = pd.DataFrame({
                'Station_ID': np.random.choice([f'ST{i:03d}' for i in range(1, n_stations+1)], n_samples),
                'Watershed_Name': np.random.choice([f'Watershed_{i}' for i in range(1, n_watersheds+1)], n_samples),
                'Water_Body_Name': np.random.choice(['River A', 'Creek B', 'Stream C', 'Lake D'], n_samples),
                'Sample_Date': pd.date_range(start='2020-01-01', end='2024-12-31', periods=n_samples),
                'Latitude': np.random.uniform(40.5, 42.5, n_samples),
                'Longitude': np.random.uniform(-74.5, -71.5, n_samples),
                'Dissolved_Oxygen_Numeric': np.random.normal(7.5, 2.0, n_samples).clip(0, 15),
                'pH_Level_Numeric': np.random.normal(7.0, 0.8, n_samples).clip(5, 9),
                'Temperature_C_Numeric': np.random.normal(15, 5, n_samples).clip(0, 30),
                'Specific_Conductivity_Numeric': np.random.lognormal(5.5, 0.8, n_samples).clip(50, 2000),
                'Total_Dissolved_Solids_Numeric': np.random.lognormal(5.3, 0.7, n_samples).clip(30, 1500),
                'Turbidity_NTU_Numeric': np.random.lognormal(2.0, 1.0, n_samples).clip(0, 100)
            })
    
    # Convert date columns
    df['Sample_Date'] = pd.to_datetime(df['Sample_Date'], errors='coerce')
    
    # Add temporal features
    df['Year'] = df['Sample_Date'].dt.year
    df['Month'] = df['Sample_Date'].dt.month
    df['Month_Name'] = df['Sample_Date'].dt.strftime('%B')
    df['Season'] = df['Sample_Date'].dt.month%12 // 3 + 1
    df['Season_Name'] = df['Season'].map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'})
    
    return df

def calculate_stress_metrics(df, conductivity_threshold, tds_threshold):
    """Calculate stormwater stress metrics for each station."""
    stress_metrics = df.groupby('Station_ID').agg({
        'Specific_Conductivity_Numeric': ['mean', 'max', 'std', 'count'],
        'Total_Dissolved_Solids_Numeric': ['mean', 'max', 'std'],
        'Latitude': 'first',
        'Longitude': 'first',
        'Watershed_Name': 'first'
    }).round(2)
    
    # Flatten column names
    stress_metrics.columns = ['_'.join(col).strip() for col in stress_metrics.columns.values]
    stress_metrics = stress_metrics.reset_index()
    
    # Calculate stress scores
    stress_metrics['conductivity_stress'] = (stress_metrics['Specific_Conductivity_Numeric_mean'] / conductivity_threshold).clip(0, 2)
    stress_metrics['tds_stress'] = (stress_metrics['Total_Dissolved_Solids_Numeric_mean'] / tds_threshold).clip(0, 2)
    stress_metrics['variability_stress'] = (stress_metrics['Specific_Conductivity_Numeric_std'] / 
                                           stress_metrics['Specific_Conductivity_Numeric_mean']).fillna(0).clip(0, 2)
    stress_metrics['stormwater_stress_score'] = (
        stress_metrics['conductivity_stress'] * 0.4 +
        stress_metrics['tds_stress'] * 0.4 +
        stress_metrics['variability_stress'] * 0.2
    ) * 100
    
    return stress_metrics

def calculate_instability_metrics(df, do_threshold, ph_min, ph_max):
    """Calculate water quality instability metrics."""
    # Calculate basic statistics
    instability_metrics = df.groupby('Station_ID').agg({
        'Dissolved_Oxygen_Numeric': ['mean', 'std', 'min', 'max'],
        'pH_Level_Numeric': ['mean', 'std', 'min', 'max'],
        'Temperature_C_Numeric': ['mean', 'std']
    }).round(2)
    
    # Flatten column names
    instability_metrics.columns = ['_'.join(col).strip() for col in instability_metrics.columns.values]
    instability_metrics = instability_metrics.reset_index()
    
    # Calculate critical events separately
    do_critical = df.groupby('Station_ID')['Dissolved_Oxygen_Numeric'].apply(
        lambda x: (x < do_threshold).sum()
    ).reset_index(name='do_critical')
    
    ph_violations = df.groupby('Station_ID')['pH_Level_Numeric'].apply(
        lambda x: ((x < ph_min) | (x > ph_max)).sum()
    ).reset_index(name='ph_violations')
    
    # Merge critical events
    instability_metrics = instability_metrics.merge(do_critical, on='Station_ID')
    instability_metrics = instability_metrics.merge(ph_violations, on='Station_ID')
    
    # Calculate coefficient of variation
    instability_metrics['do_cv'] = (instability_metrics['Dissolved_Oxygen_Numeric_std'] / 
                                    instability_metrics['Dissolved_Oxygen_Numeric_mean']).fillna(0)
    instability_metrics['ph_cv'] = (instability_metrics['pH_Level_Numeric_std'] / 
                                   instability_metrics['pH_Level_Numeric_mean']).fillna(0)
    
    # Get sample counts for normalization
    sample_counts = df.groupby('Station_ID').size().reset_index(name='sample_count')
    instability_metrics = instability_metrics.merge(sample_counts, on='Station_ID')
    
    # Calculate instability score
    instability_metrics['instability_score'] = (
        instability_metrics['do_cv'] * 50 +
        instability_metrics['ph_cv'] * 30 +
        (instability_metrics['do_critical'] / instability_metrics['sample_count']).fillna(0) * 20
    ) * 100
    
    return instability_metrics

def identify_hotspots(stress_metrics, instability_metrics, df):
    """Identify conservation hotspots based on multiple metrics."""
    # Combine metrics
    hotspots = stress_metrics[['Station_ID', 'Latitude_first', 'Longitude_first', 
                               'stormwater_stress_score', 'Watershed_Name_first']].copy()
    hotspots.columns = ['Station_ID', 'Latitude', 'Longitude', 'stress_score', 'Watershed']
    
    # Add instability score - select only the columns we need
    instability_cols = ['Station_ID', 'instability_score']
    if 'do_critical' in instability_metrics.columns:
        instability_cols.append('do_critical')
    if 'ph_violations' in instability_metrics.columns:
        instability_cols.append('ph_violations')
    
    hotspots = hotspots.merge(
        instability_metrics[instability_cols],
        on='Station_ID', how='left'
    )
    
    # Calculate persistence score
    problem_counts = df.groupby('Station_ID', group_keys=False).apply(
        lambda x: pd.Series({
            'total_samples': len(x),
            'problem_events': (
                (x['Dissolved_Oxygen_Numeric'] < 5).sum() +
                ((x['pH_Level_Numeric'] < 6.5) | (x['pH_Level_Numeric'] > 8.5)).sum() +
                (x['Specific_Conductivity_Numeric'] > 500).sum()
            )
        }), include_groups=False
    ).reset_index()
    
    problem_counts['persistence_score'] = (problem_counts['problem_events'] / 
                                          problem_counts['total_samples'] * 100).round(1)
    
    hotspots = hotspots.merge(problem_counts[['Station_ID', 'persistence_score']], 
                             on='Station_ID', how='left')
    
    # Calculate final hotspot score
    hotspots['hotspot_score'] = (
        hotspots['stress_score'].fillna(0) * 0.35 +
        hotspots['instability_score'].fillna(0) * 0.35 +
        hotspots['persistence_score'].fillna(0) * 0.30
    ).round(1)
    
    # Assign priority levels - handle NaN values
    hotspots['priority'] = pd.cut(
        hotspots['hotspot_score'],
        bins=[0, 25, 50, 75, float('inf')],  # Use inf instead of 100 to catch all high values
        labels=['Low', 'Moderate', 'High', 'Critical'],
        include_lowest=True
    )
    
    # Fill any missing columns with default values for compatibility
    if 'do_critical' not in hotspots.columns:
        hotspots['do_critical'] = 0
    if 'ph_violations' not in hotspots.columns:
        hotspots['ph_violations'] = 0
    
    return hotspots

def create_overview_metrics(df, hotspots):
    """Create overview metrics for the dashboard."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", f"{len(df):,}")
        st.metric("Date Range", f"{df['Sample_Date'].min().date()} to {df['Sample_Date'].max().date()}")
    
    with col2:
        st.metric("Monitoring Stations", df['Station_ID'].nunique())
        st.metric("Watersheds", df['Watershed_Name'].nunique())
    
    with col3:
        critical_stations = len(hotspots[hotspots['priority'] == 'Critical'])
        st.metric("Critical Priority Stations", critical_stations)
        st.caption(f"{critical_stations/len(hotspots)*100:.1f}% of total")
    
    with col4:
        avg_do = df['Dissolved_Oxygen_Numeric'].mean()
        st.metric("Average DO", f"{avg_do:.2f} mg/L",
                 delta="Above threshold" if avg_do > 5 else "Below threshold",
                 delta_color="normal" if avg_do > 5 else "inverse")

def create_parameter_distributions(df, do_threshold, ph_min, ph_max):
    """Create parameter distribution plots."""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Dissolved Oxygen',
            'pH Level',
            'Temperature',
            'Conductivity',
            'TDS',
            'Turbidity'
        )
    )
    
    params = [
        ('Dissolved_Oxygen_Numeric', 'DO (mg/L)', 1, 1, do_threshold, None),
        ('pH_Level_Numeric', 'pH', 1, 2, ph_min, ph_max),
        ('Temperature_C_Numeric', 'Temperature (C)', 1, 3, None, None),
        ('Specific_Conductivity_Numeric', 'Conductivity (S/cm)', 2, 1, None, 500),
        ('Total_Dissolved_Solids_Numeric', 'TDS (mg/L)', 2, 2, None, 500),
        ('Turbidity_NTU_Numeric', 'Turbidity (NTU)', 2, 3, None, None)
    ]
    
    for param, label, row, col, threshold_min, threshold_max in params:
        if param in df.columns:
            fig.add_trace(
                go.Histogram(
                    x=df[param],
                    name=label,
                    marker_color='lightblue',
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Add threshold lines
            if threshold_min:
                fig.add_vline(x=threshold_min, line_dash="dash", line_color="red",
                            annotation_text=f"Min: {threshold_min}", row=row, col=col)
            if threshold_max:
                fig.add_vline(x=threshold_max, line_dash="dash", line_color="red",
                            annotation_text=f"Max: {threshold_max}", row=row, col=col)
    
    fig.update_layout(height=600, showlegend=False, title_text="<b>Water Quality Parameter Distributions</b>")
    return fig

def create_seasonal_analysis(df):
    """Create seasonal analysis plots."""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'DO by Season',
            'pH by Season',
            'Temperature by Season',
            'Monthly DO Trends',
            'Monthly Conductivity',
            'Yearly Parameter Trends'
        ),
        specs=[[{'type': 'box'}, {'type': 'box'}, {'type': 'box'}],
               [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    # Seasonal box plots
    for season in df['Season_Name'].unique():
        season_data = df[df['Season_Name'] == season]
        
        fig.add_trace(go.Box(y=season_data['Dissolved_Oxygen_Numeric'], name=season), row=1, col=1)
        fig.add_trace(go.Box(y=season_data['pH_Level_Numeric'], name=season), row=1, col=2)
        fig.add_trace(go.Box(y=season_data['Temperature_C_Numeric'], name=season), row=1, col=3)
    
    # Monthly trends
    monthly_do = df.groupby('Month')['Dissolved_Oxygen_Numeric'].mean()
    monthly_cond = df.groupby('Month')['Specific_Conductivity_Numeric'].mean()
    
    fig.add_trace(
        go.Scatter(x=list(range(1, 13)), y=monthly_do, mode='lines+markers', name='DO'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=list(range(1, 13)), y=monthly_cond, mode='lines+markers', name='Conductivity'),
        row=2, col=2
    )
    
    # Yearly trends
    yearly_data = df.groupby('Year').agg({
        'Dissolved_Oxygen_Numeric': 'mean',
        'pH_Level_Numeric': 'mean',
        'Temperature_C_Numeric': 'mean'
    })
    
    for param in yearly_data.columns:
        fig.add_trace(
            go.Scatter(x=yearly_data.index, y=yearly_data[param], mode='lines+markers', name=param),
            row=2, col=3
        )
    
    fig.update_layout(height=700, title_text="<b>Seasonal and Temporal Analysis</b>", showlegend=False)
    return fig

def create_hotspot_map(hotspots):
    """Create an interactive map of water quality hotspots."""
    fig = px.scatter_mapbox(
        hotspots,
        lat='Latitude',
        lon='Longitude',
        color='hotspot_score',
        size='hotspot_score',
        color_continuous_scale='RdYlGn_r',
        size_max=15,
        zoom=7,
        mapbox_style='carto-positron',
        title='<b>Water Quality Hotspots - Conservation Priority Map</b>',
        hover_name='Station_ID',
        hover_data={
            'priority': True,
            'stress_score': ':.2f',
            'instability_score': ':.2f',
            'persistence_score': ':.2f',
            'hotspot_score': ':.1f',
            'Watershed': True,
            'Latitude': False,
            'Longitude': False
        },
        labels={'hotspot_score': 'Hotspot Score'},
        height=700
    )
    
    fig.update_layout(
        mapbox=dict(
            center=dict(
                lat=hotspots['Latitude'].mean(),
                lon=hotspots['Longitude'].mean()
            )
        )
    )
    
    return fig

def create_stress_analysis(stress_metrics):
    """Create stormwater stress analysis visualizations."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Conductivity vs TDS Correlation',
            'Stress Score Distribution',
            'Top 15 Stressed Stations',
            'Stress by Watershed'
        ),
        specs=[[{'type': 'scatter'}, {'type': 'histogram'}],
               [{'type': 'bar'}, {'type': 'box'}]]
    )
    
    # Conductivity vs TDS scatter
    fig.add_trace(
        go.Scatter(
            x=stress_metrics['Specific_Conductivity_Numeric_mean'],
            y=stress_metrics['Total_Dissolved_Solids_Numeric_mean'],
            mode='markers',
            marker=dict(
                size=8,
                color=stress_metrics['stormwater_stress_score'],
                colorscale='Reds',
                showscale=True
            ),
            text=stress_metrics['Station_ID'],
            hovertemplate='Station: %{text}<br>Conductivity: %{x:.1f}<br>TDS: %{y:.1f}'
        ),
        row=1, col=1
    )
    
    # Stress score distribution
    fig.add_trace(
        go.Histogram(
            x=stress_metrics['stormwater_stress_score'],
            nbinsx=30,
            marker_color='coral'
        ),
        row=1, col=2
    )
    
    # Top stressed stations
    top_stressed = stress_metrics.nlargest(15, 'stormwater_stress_score')
    fig.add_trace(
        go.Bar(
            x=top_stressed['stormwater_stress_score'],
            y=top_stressed['Station_ID'],
            orientation='h',
            marker_color='darkred'
        ),
        row=2, col=1
    )
    
    # Stress by watershed
    watershed_stress = stress_metrics.groupby('Watershed_Name_first')['stormwater_stress_score'].apply(list)
    for watershed, scores in watershed_stress.items():
        fig.add_trace(
            go.Box(y=scores, name=watershed[:15]),
            row=2, col=2
        )
    
    fig.update_layout(height=800, title_text="<b>Stormwater Stress Analysis</b>", showlegend=False)
    return fig

def create_instability_analysis(instability_metrics):
    """Create water quality instability visualizations."""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'DO Variability (CV)',
            'pH Variability (CV)',
            'DO vs pH Instability',
            'Critical DO Events',
            'pH Violations',
            'Instability Score Distribution'
        )
    )
    
    # DO variability
    fig.add_trace(
        go.Histogram(x=instability_metrics['do_cv'], nbinsx=30, marker_color='lightblue'),
        row=1, col=1
    )
    
    # pH variability
    fig.add_trace(
        go.Histogram(x=instability_metrics['ph_cv'], nbinsx=30, marker_color='lightgreen'),
        row=1, col=2
    )
    
    # DO vs pH instability scatter
    fig.add_trace(
        go.Scatter(
            x=instability_metrics['do_cv'],
            y=instability_metrics['ph_cv'],
            mode='markers',
            marker=dict(
                size=8,
                color=instability_metrics['instability_score'],
                colorscale='Viridis',
                showscale=True
            ),
            text=instability_metrics['Station_ID']
        ),
        row=1, col=3
    )
    
    # Critical DO events
    fig.add_trace(
        go.Histogram(x=instability_metrics['do_critical'], nbinsx=20, marker_color='red'),
        row=2, col=1
    )
    
    # pH violations
    fig.add_trace(
        go.Histogram(x=instability_metrics['ph_violations'], nbinsx=20, marker_color='orange'),
        row=2, col=2
    )
    
    # Instability score distribution
    fig.add_trace(
        go.Histogram(x=instability_metrics['instability_score'], nbinsx=30, marker_color='purple'),
        row=2, col=3
    )
    
    fig.update_layout(height=600, title_text="<b>Water Quality Instability Analysis</b>", showlegend=False)
    return fig

def create_comprehensive_dashboard(hotspots, df):
    """Create a comprehensive dashboard with multiple metrics."""
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            'Priority Distribution',
            'Top 10 Hotspots',
            'Score Components',
            'Hotspots by Watershed',
            'Stress vs Instability',
            'Problem Events Distribution',
            'Parameter Correlations',
            'Monthly Violations',
            'Action Priority Matrix'
        ),
        specs=[[{'type': 'pie'}, {'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'box'}, {'type': 'scatter'}, {'type': 'histogram'}],
               [{'type': 'heatmap'}, {'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    # 1. Priority distribution pie
    priority_counts = hotspots['priority'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=priority_counts.index,
            values=priority_counts.values,
            marker=dict(colors=['green', 'yellow', 'orange', 'red'])
        ),
        row=1, col=1
    )
    
    # 2. Top 10 hotspots
    top10 = hotspots.nlargest(10, 'hotspot_score')
    fig.add_trace(
        go.Bar(
            x=top10['hotspot_score'],
            y=top10['Station_ID'],
            orientation='h',
            marker_color='darkred'
        ),
        row=1, col=2
    )
    
    # 3. Score components
    score_means = hotspots[['stress_score', 'instability_score', 'persistence_score']].mean()
    fig.add_trace(
        go.Bar(
            x=['Stress', 'Instability', 'Persistence'],
            y=score_means.values,
            marker_color=['coral', 'lightblue', 'lightgreen']
        ),
        row=1, col=3
    )
    
    # 4. Hotspots by watershed
    for watershed in hotspots['Watershed'].unique()[:5]:
        watershed_data = hotspots[hotspots['Watershed'] == watershed]
        fig.add_trace(
            go.Box(y=watershed_data['hotspot_score'], name=watershed[:10]),
            row=2, col=1
        )
    
    # 5. Stress vs Instability
    fig.add_trace(
        go.Scatter(
            x=hotspots['stress_score'],
            y=hotspots['instability_score'],
            mode='markers',
            marker=dict(
                size=hotspots['persistence_score']/5,
                color=hotspots['hotspot_score'],
                colorscale='RdYlGn_r',
                showscale=True
            ),
            text=hotspots['Station_ID']
        ),
        row=2, col=2
    )
    
    # Add more visualizations as needed...
    
    fig.update_layout(height=1000, title_text="<b>Comprehensive Water Quality Dashboard</b>", showlegend=False)
    return fig

def main():
    """Main application function."""
    
    # Load data - prioritize sample_water_quality.csv
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.success(f"Loaded {len(df):,} rows from uploaded file")
    elif use_sample:
        df = load_data()  # Will use sample_water_quality.csv by default
        # Show info about loaded dataset
        import os
        if os.path.exists('sample_water_quality.csv'):
            st.success(f"Loaded {len(df):,} rows from sample_water_quality.csv")
        else:
            st.warning("Using generated sample data. sample_water_quality.csv not found.")
    else:
        st.warning("Please upload a CSV file or select 'Use Sample Data' to proceed.")
        return
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Overview", 
        "Parameters", 
        "Seasonal", 
        "Stress Analysis",
        "Hotspots",
        "Dashboard",
        " WSLUS Model Insights"
    ])
    
    # Calculate metrics
    stress_metrics = calculate_stress_metrics(df, conductivity_max, tds_max)
    instability_metrics = calculate_instability_metrics(df, do_threshold, ph_min, ph_max)
    hotspots = identify_hotspots(stress_metrics, instability_metrics, df)
    
    with tab1:
        st.header("Data Overview")
        create_overview_metrics(df, hotspots)
        
        st.markdown("---")
        
        # Data summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Statistics")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.subheader("Top Priority Stations")
            top_priorities = hotspots.nlargest(10, 'hotspot_score')[
                ['Station_ID', 'Watershed', 'priority', 'hotspot_score']
            ]
            st.dataframe(top_priorities, use_container_width=True)
        
        # Timeline
        st.subheader("Sampling Timeline")
        timeline_fig = px.histogram(
            df, x='Sample_Date', 
            title='Sample Collection Over Time',
            labels={'count': 'Number of Samples'}
        )
        st.plotly_chart(timeline_fig, use_container_width=True)
    
    with tab2:
        st.header("Water Quality Parameters Analysis")
        
        # Parameter distributions
        dist_fig = create_parameter_distributions(df, do_threshold, ph_min, ph_max)
        st.plotly_chart(dist_fig, use_container_width=True)
        
        # Correlation matrix
        st.subheader("Parameter Correlations")
        numeric_cols = ['Dissolved_Oxygen_Numeric', 'pH_Level_Numeric', 'Temperature_C_Numeric',
                       'Specific_Conductivity_Numeric', 'Total_Dissolved_Solids_Numeric']
        
        if all(col in df.columns for col in numeric_cols):
            corr_matrix = df[numeric_cols].corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig_corr.update_layout(
                title="<b>Parameter Correlation Matrix</b>",
                height=500
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab3:
        st.header("Seasonal and Temporal Analysis")
        
        seasonal_fig = create_seasonal_analysis(df)
        st.plotly_chart(seasonal_fig, use_container_width=True)
        
        # Additional seasonal insights
        st.subheader("Seasonal Statistics")
        
        seasonal_stats = df.groupby('Season_Name').agg({
            'Dissolved_Oxygen_Numeric': ['mean', 'std'],
            'pH_Level_Numeric': ['mean', 'std'],
            'Temperature_C_Numeric': ['mean', 'std']
        }).round(2)
        
        st.dataframe(seasonal_stats, use_container_width=True)
    
    with tab4:
        st.header("Stormwater Stress Analysis")
        
        # Stress analysis plots
        stress_fig = create_stress_analysis(stress_metrics)
        st.plotly_chart(stress_fig, use_container_width=True)
        
        # Instability analysis
        st.subheader("Water Quality Instability")
        instability_fig = create_instability_analysis(instability_metrics)
        st.plotly_chart(instability_fig, use_container_width=True)
    
    with tab5:
        st.header("Conservation Hotspot Identification")
        
        # Interactive map
        map_fig = create_hotspot_map(hotspots)
        st.plotly_chart(map_fig, use_container_width=True)
        
        # Hotspot statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Critical Priority", 
                     len(hotspots[hotspots['priority'] == 'Critical']),
                     help="Stations requiring immediate attention")
        
        with col2:
            st.metric("High Priority",
                     len(hotspots[hotspots['priority'] == 'High']),
                     help="Stations requiring attention within 3 months")
        
        with col3:
            st.metric("Average Hotspot Score",
                     f"{hotspots['hotspot_score'].mean():.1f}",
                     help="Overall watershed health indicator")
        
        # Priority recommendations
        st.subheader("Conservation Recommendations")
        
        critical_stations = hotspots[hotspots['priority'] == 'Critical']
        if len(critical_stations) > 0:
            st.warning(f"**{len(critical_stations)} stations require immediate intervention:**")
            
            for _, station in critical_stations.iterrows():
                st.write(f"**{station['Station_ID']}** (Watershed: {station['Watershed']})")
                st.write(f"  - Hotspot Score: {station['hotspot_score']:.1f}")
                st.write(f"  - Main Issues: Stress={station['stress_score']:.1f}, "
                        f"Instability={station['instability_score']:.1f}, "
                        f"Persistence={station['persistence_score']:.1f}")
    
    with tab6:
        st.header("Comprehensive Dashboard")
        
        dashboard_fig = create_comprehensive_dashboard(hotspots, df)
        st.plotly_chart(dashboard_fig, use_container_width=True)
        
    with tab7:
        create_wslus_insights_tab()

def load_wslus_model_results():
    """Load WSLUS model results if available"""
    try:
        import geopandas as gpd
        import joblib
        
        # Try to load the model results
        stations_df = None
        model = None
        feature_importance = None
        
        # Load stations with WSLUS scores
        try:
            stations_df = pd.read_csv('stations_with_wslus_scores.csv')
            st.success("Loaded WSLUS model results")
        except:
            pass
        
        # Load model if available
        try:
            model = joblib.load('wslus_predictive_model.pkl')
        except:
            pass
        
        return stations_df, model, feature_importance
    except Exception as e:
        return None, None, None

def create_wslus_insights_tab():
    """Create high-level, visualization-heavy WSLUS insights for government stakeholders"""
    
    # Hero section
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); padding: 30px; border-radius: 10px; margin-bottom: 30px;">
        <h1 style="color: white; margin: 0;">Watershed Preservation Opportunity Map</h1>
        <p style="color: #e0e7ff; font-size: 18px; margin: 10px 0 0 0;">
            Data-driven insights to prioritize land conservation and improve water quality
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Try to load model results
    stations_df, model, feature_importance = load_wslus_model_results()
    
    if stations_df is None or len(stations_df) == 0:
        st.warning(""" ** Data Required**
        
        Please run the `watershed_preservation_model.ipynb` notebook to generate analysis results.
        The notebook will create the necessary files for this dashboard.
        """)
        return
    
    # Convert to numeric for WSLUS if needed
    if 'WSLUS' in stations_df.columns:
        stations_df['WSLUS'] = pd.to_numeric(stations_df['WSLUS'], errors='coerce')
    
    # ========== EXECUTIVE SUMMARY ==========
    st.markdown("## Executive Summary")
    
    # Key metrics in large cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        critical = len(stations_df[stations_df['WSLUS'] > 75]) if 'WSLUS' in stations_df.columns else 0
        st.metric("Critical Areas", critical)
        st.caption(f"{critical/len(stations_df)*100:.1f}% of stations | WSLUS Score > 75" if len(stations_df) > 0 else "WSLUS Score > 75")
    
    with col2:
        high_risk = len(stations_df[(stations_df['WSLUS'] > 50) & (stations_df['WSLUS'] <= 75)]) if 'WSLUS' in stations_df.columns else 0
        st.metric("High Priority", high_risk)
        st.caption(f"{high_risk/len(stations_df)*100:.1f}% of stations | WSLUS Score 50-75" if len(stations_df) > 0 else "WSLUS Score 50-75")
    
    with col3:
        ej_critical = len(stations_df[(stations_df.get('in_ej_community', 0) == 1) & 
                                     (stations_df.get('WSLUS', 0) > 75)]) if 'WSLUS' in stations_df.columns else 0
        st.metric(
            "EJ Communities at Risk", 
            ej_critical,
            delta="Requires immediate action" if ej_critical > 0 else "None identified",
            delta_color="inverse" if ej_critical > 0 else "normal"
        )
        st.caption("Environmental Justice areas")
    
    with col4:
        avg_wslus = stations_df['WSLUS'].mean() if 'WSLUS' in stations_df.columns else 0
        st.metric(
            "Average Water Stress", 
            f"{avg_wslus:.1f}",
            delta="Target: <50" if avg_wslus > 50 else "Within target",
            delta_color="inverse" if avg_wslus > 50 else "normal"
        )
        st.caption("WSLUS Score (0-100)")
    
        st.markdown("---")
    
    # ========== HOW WE CALCULATE WATER STRESS (WSLUS) ==========
    st.markdown("## Understanding WSLUS: Water Stress from Land Use Score")
    
    # Introduction section
    st.markdown("""
    **WSLUS (Water Stress from Land Use Score)** is a composite metric that quantifies the overall water quality stress at a monitoring station. 
    It combines multiple water quality parameters into a single score (0-100) that indicates how stressed the water system is.
    
    **Why WSLUS?**
    - **Single Metric**: Instead of tracking 5+ separate water quality parameters, WSLUS provides one clear number
    - **Prioritization**: Helps identify which areas need immediate attention
    - **Comparability**: Allows direct comparison between different monitoring stations
    - **Actionable**: Clear risk categories guide conservation and intervention decisions
    """)
    
    with st.expander("Detailed WSLUS Methodology", expanded=True):
        st.markdown("""
        ### What is WSLUS?
        
        WSLUS stands for **Water Stress from Land Use Score**. It's a weighted composite score that measures overall water quality stress 
        by combining five key water quality parameters. The score ranges from 0 (healthy) to 100 (critically stressed).
        
        ### Formula
        
        The **Water Stress from Land Use Score (WSLUS)** is calculated from five water quality parameters:
        
        ```
        WSLUS = (Conductivity Stress × 0.30) + 
                (TDS Stress × 0.25) + 
                (pH Stress × 0.20) + 
                (Temperature Stress × 0.15) + 
                (DO Stress × 0.10)
        ```
        
        ### Component Calculations
        
        Each stress component is normalized to a 0-1 scale based on water quality thresholds:
        
        **1. Conductivity Stress (30% weight)**
        - Measures dissolved ions (road salt, runoff, pollution)
        - High conductivity (>500 μS/cm) indicates contamination
        - Formula: `min(1.0, (conductivity - 200) / 500)`
        
        **2. TDS Stress (25% weight)**
        - Total Dissolved Solids (related to conductivity)
        - High TDS (>500 mg/L) indicates poor water quality
        - Formula: `min(1.0, (TDS - 200) / 500)`
        
        **3. pH Stress (20% weight)**
        - Measures acidity/alkalinity (optimal range: 6.5-8.5)
        - Deviations from neutral indicate stress
        - Formula: `max(0, abs(pH - 7.0) - 1.5) / 2.5`
        
        **4. Temperature Stress (15% weight)**
        - Thermal pollution affects aquatic ecosystems
        - Optimal: 10-20°C for most freshwater systems
        - Formula: `max(0, (temperature - 15) / 10)`
        
        **5. DO Stress (10% weight)**
        - Dissolved Oxygen (low = poor water quality)
        - Critical threshold: <5 mg/L
        - Formula: `max(0, (5 - DO) / 5)`
        
        ### EJ Amplification
        
        **Environmental Justice Boost**: Stations in EJ communities have their WSLUS scores multiplied by **1.5x** to prioritize areas where environmental benefits will have the greatest social impact.
        
        ### Risk Categories
        
        WSLUS scores are categorized into four risk levels:
        
        - **Low (0-25)**: Healthy water quality, minimal intervention needed
          - Water quality parameters are within acceptable ranges
          - Ecosystem is functioning well
          - Continue monitoring, no immediate action required
        
        - **Moderate (25-50)**: Some concerns, monitoring recommended
          - Some water quality parameters show signs of stress
          - Ecosystem may be under pressure
          - Increased monitoring frequency recommended
          - Consider preventive measures
        
        - **High (50-75)**: Significant stress, intervention needed
          - Multiple water quality parameters indicate problems
          - Ecosystem health is compromised
          - Active intervention recommended
          - Conservation efforts should be prioritized
        
        - **Critical (75-100)**: Immediate action required
          - Severe water quality degradation
          - Ecosystem is at risk
          - Urgent intervention required
          - Highest priority for conservation resources
        
        ### How to Interpret WSLUS Scores
        
        **Example Scenarios:**
        
        - **WSLUS = 20 (Low Risk)**: All water quality parameters are healthy. The area is functioning well ecologically.
        
        - **WSLUS = 45 (Moderate Risk)**: Some parameters (e.g., slightly elevated conductivity) indicate stress. 
          Monitor closely and consider preventive conservation.
        
        - **WSLUS = 65 (High Risk)**: Multiple parameters show problems (e.g., high conductivity, low DO, pH imbalance). 
          This area needs active intervention and conservation efforts.
        
        - **WSLUS = 85 (Critical Risk)**: Severe water quality issues across multiple parameters. 
          Immediate conservation action is required to protect the ecosystem.
        
        ### Why These Weights?
        
        The component weights (30% conductivity, 25% TDS, 20% pH, 15% temperature, 10% DO) reflect:
        
        - **Conductivity/TDS (55% combined)**: Most common indicators of pollution and runoff
        - **pH (20%)**: Critical for aquatic life, but less variable than conductivity
        - **Temperature (15%)**: Important for ecosystem health, but often secondary to chemical parameters
        - **DO (10%)**: Essential but often correlated with other parameters
        
        ### Real-World Application
        
        WSLUS helps answer key questions:
        - **Where should we focus conservation efforts?** → High/Critical WSLUS areas
        - **Which areas are most at risk?** → Compare WSLUS scores across stations
        - **Is our intervention working?** → Track WSLUS changes over time
        - **Where will conservation have the most impact?** → Combine WSLUS with economic vulnerability (Intervention Priority)
        """)
    
    # Summary stats for risk distribution
    if 'WSLUS' in stations_df.columns and 'risk_category' in stations_df.columns:
        st.markdown("### Current Risk Distribution")
        risk_counts = stations_df['risk_category'].value_counts()
        col1, col2, col3, col4 = st.columns(4)
        
        total = len(stations_df)
        with col1:
            low = risk_counts.get('Low', 0)
            st.metric("Low Risk", f"{low}")
            st.caption(f"{low/total*100:.1f}% of stations")
        with col2:
            moderate = risk_counts.get('Moderate', 0)
            st.metric("Moderate Risk", f"{moderate}")
            st.caption(f"{moderate/total*100:.1f}% of stations")
        with col3:
            high = risk_counts.get('High', 0)
            st.metric("High Risk", f"{high}")
            st.caption(f"{high/total*100:.1f}% of stations")
        with col4:
            critical = risk_counts.get('Critical', 0)
            st.metric("Critical Risk", f"{critical}")
            st.caption(f"{critical/total*100:.1f}% of stations")
    
    st.markdown("---")
    
    # ========== HOW WE IDENTIFY HIGH STRESS AREAS ==========
    st.markdown("## How We Identify High Stress Areas")
    
    with st.expander("High Stress Area Identification Methodology", expanded=True):
        st.markdown("""
        ### Threshold Determination
        
        High stress areas are identified using an **adaptive threshold** approach:
        
        **Primary Method**: 75th Percentile
        - Calculate the 75th percentile of all WSLUS scores
        - Any station with WSLUS above this threshold is considered "high stress"
        - This ensures we capture the top 25% of stressed areas
        
        **Safety Threshold**: Minimum of 40
        - If the 75th percentile is below 40, use 40 as the threshold
        - This ensures we don't miss areas with moderate-to-high stress
        - Formula: `threshold = max(40, percentile_75(WSLUS))`
        
        ### Classification Process
        
        1. **Calculate WSLUS** for all monitoring stations using the formula above
        2. **Apply EJ amplification** (1.5x multiplier) for EJ communities
        3. **Determine threshold** using adaptive method
        4. **Classify stations**:
           - High Stress: WSLUS ≥ threshold
           - Low Stress: WSLUS < threshold
        
        ### Why This Approach?
        
        - **Adaptive**: Adjusts to the actual distribution of water stress in the dataset
        - **Robust**: Works even when most stations have low stress (uses minimum threshold)
        - **Prioritized**: Focuses on the most stressed areas relative to the dataset
        - **EJ-Aware**: Automatically prioritizes EJ communities through score amplification
        
        ### Example
        
        If the dataset has WSLUS scores ranging from 20 to 80:
        - 75th percentile = 65
        - Threshold = max(40, 65) = **65**
        - All stations with WSLUS ≥ 65 are classified as "high stress"
        """)
    
    # Show current high stress identification
    if 'WSLUS' in stations_df.columns:
        wslus_scores = stations_df['WSLUS'].dropna()
        if len(wslus_scores) > 0:
            threshold_75 = wslus_scores.quantile(0.75)
            threshold_used = max(40, threshold_75)
            high_stress_count = len(stations_df[stations_df['WSLUS'] >= threshold_used])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("75th Percentile", f"{threshold_75:.1f}", 
                         delta="WSLUS score")
            with col2:
                st.metric("Threshold Used", f"{threshold_used:.1f}",
                         delta="Adaptive threshold")
            with col3:
                st.metric("High Stress Stations", f"{high_stress_count}",
                         delta=f"{high_stress_count/len(stations_df)*100:.1f}%")
    
    st.markdown("---")
    
    # ========== ENVIRONMENTAL JUSTICE INFORMATION ==========
    st.markdown("## About Environmental Justice (EJ)")
    
    with st.expander("What is Environmental Justice?", expanded=False):
        st.markdown("""
        **Environmental Justice (EJ)** refers to the fair treatment and meaningful involvement of all people regardless of race, color, national origin, or income with respect to the development, implementation, and enforcement of environmental laws, regulations, and policies.
        
        ### Key Principles:
        - **Fair Treatment**: No group should bear a disproportionate share of negative environmental consequences
        - **Meaningful Involvement**: All people should have equal opportunity to participate in environmental decision-making
        - **Equal Protection**: Environmental laws should be applied equally across all communities
        
        ### Why EJ Matters for Watershed Conservation:
        - Low-income and minority communities often face higher exposure to environmental hazards
        - These communities may have less access to clean water and green spaces
        - Conservation efforts should prioritize areas where environmental benefits will have the greatest social impact
        - Addressing environmental inequities is both a moral imperative and a public health priority
        """)
    
    with st.expander("How We Identify EJ Communities", expanded=False):
        st.markdown("""
        ### Data Sources and Methodology
        
        **Primary Indicator: Median Household Income**
        - Communities with median household income **below $50,000** are classified as EJ communities
        - This threshold aligns with Massachusetts state EJ designation criteria
        - Income data comes from **Massachusetts Department of Revenue (DOR) Municipal Databank**
        
        **Additional Factors Considered:**
        - **Poverty Rate**: Areas with higher poverty rates (>15%) are prioritized
        - **Population Density**: Urban areas with high density and low income are flagged
        - **Economic Vulnerability**: Composite score based on income, poverty, and education levels
        
        ### Data Integration Process:
        1. **Municipality Matching**: Stations are matched to municipalities using:
           - Location descriptions (e.g., "Wellfleet", "Seekonk")
           - Geographic coordinates (reverse geocoding)
        
        2. **Economic Data**: 
           - Real income data from MA DOR (when available)
           - Location-based proxies for unmatched stations
           - Station-specific variation to ensure realistic diversity
        
        3. **EJ Classification**:
           - Binary flag: `in_ej_community` (1 = EJ, 0 = not EJ)
           - Vulnerability score: `ej_vulnerability` (0-1 scale)
           - Used to amplify WSLUS scores by 1.5x for EJ areas
        
        ### Limitations:
        - Municipality matching may not capture all stations
        - Income thresholds are approximate (actual MA EJ criteria may vary)
        - For production use, integrate with official MassGIS EJ 2020 Census data
        """)
    
    st.markdown("---")
    
    # ========== NEW: Economic Data Integration ==========
    st.markdown("## Economic & Demographic Context")
    
    # Check for economic indicators
    economic_cols = [c for c in stations_df.columns if any(x in c for x in ['income', 'poverty', 'density', 'college', 'ej_vulnerability', 'urbanization_proxy'])]
    
    if economic_cols:
        st.success(f"Economic data integrated: {len(economic_cols)} indicators available")
        
        # Comprehensive Economic Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'median_household_income' in stations_df.columns:
                income_data = stations_df['median_household_income'].dropna()
                if len(income_data) > 0:
                    median_income = income_data.median()
                    q25 = income_data.quantile(0.25)
                    q75 = income_data.quantile(0.75)
                    st.metric("Median Household Income", f"${median_income:,.0f}", 
                             delta=f"IQR: ${q25:,.0f} - ${q75:,.0f}")
                    st.caption(f"Range: ${income_data.min():,.0f} - ${income_data.max():,.0f}")
        
        with col2:
            if 'poverty_rate' in stations_df.columns:
                poverty_data = stations_df['poverty_rate'].dropna()
                if len(poverty_data) > 0:
                    median_poverty = poverty_data.median()
                    high_poverty = len(poverty_data[poverty_data > 15])
                    st.metric("Median Poverty Rate", f"{median_poverty:.1f}%",
                             delta=f"{high_poverty} areas >15%")
                    st.caption(f"Range: {poverty_data.min():.1f}% - {poverty_data.max():.1f}%")
        
        with col3:
            if 'population_density' in stations_df.columns:
                density_data = stations_df['population_density'].dropna()
                if len(density_data) > 0:
                    median_density = density_data.median()
                    urban_areas = len(density_data[density_data > 1000])
                    st.metric("Median Population Density", f"{median_density:,.0f}/km²",
                             delta=f"{urban_areas} urban areas")
                    st.caption(f"Range: {density_data.min():,.0f} - {density_data.max():,.0f}/km²")
        
        with col4:
            if 'in_ej_community' in stations_df.columns:
                ej_count = stations_df['in_ej_community'].sum()
                ej_pct = (ej_count / len(stations_df)) * 100
                if 'ej_vulnerability' in stations_df.columns:
                    ej_vuln = stations_df[stations_df['in_ej_community'] == 1]['ej_vulnerability'].mean()
                    st.metric("EJ Communities", f"{ej_count}")
                    if not pd.isna(ej_vuln):
                        st.caption(f"Avg vulnerability: {ej_vuln:.2f}")
                    else:
                        st.caption(f"{ej_pct:.1f}% of stations")
                else:
                    st.metric("EJ Communities", f"{ej_count}")
                    st.caption(f"{ej_pct:.1f}% of stations")
        
        # Income Distribution Analysis
        if 'median_household_income' in stations_df.columns:
            st.markdown("### Income Distribution Analysis")
            
            income_data = stations_df['median_household_income'].dropna()
            if len(income_data) > 0:
                col1, col2 = st.columns(2)
        
                with col1:
                    # Income distribution histogram
                    income_fig = go.Figure()
                    income_fig.add_trace(go.Histogram(
                        x=income_data,
                        nbinsx=20,
                        name='Household Income',
                        marker_color='#3b82f6',
                        opacity=0.7
                    ))
                    
                    # Add quartile lines
                    q25 = income_data.quantile(0.25)
                    q50 = income_data.quantile(0.50)
                    q75 = income_data.quantile(0.75)
                    
                    income_fig.add_vline(x=q25, line_dash="dash", line_color="orange", 
                                        annotation_text=f"Q1: ${q25:,.0f}")
                    income_fig.add_vline(x=q50, line_dash="dash", line_color="red", 
                                        annotation_text=f"Median: ${q50:,.0f}")
                    income_fig.add_vline(x=q75, line_dash="dash", line_color="orange", 
                                        annotation_text=f"Q3: ${q75:,.0f}")
                    
                    income_fig.update_layout(
                        title="<b>Household Income Distribution</b>",
                        xaxis_title="Median Household Income ($)",
                        yaxis_title="Number of Stations",
                        height=400
                    )
                    st.plotly_chart(income_fig, use_container_width=True)
        
        with col2:
                    # Income stratification
                    low_income_threshold = income_data.quantile(0.33)
                    high_income_threshold = income_data.quantile(0.67)
                    
                    low_income = len(income_data[income_data < low_income_threshold])
                    middle_income = len(income_data[(income_data >= low_income_threshold) & (income_data < high_income_threshold)])
                    high_income = len(income_data[income_data >= high_income_threshold])
                    
                    strat_fig = go.Figure(data=[
                        go.Bar(
                            x=['Low Income', 'Middle Income', 'High Income'],
                            y=[low_income, middle_income, high_income],
                            marker_color=['#ef4444', '#fbbf24', '#10b981'],
                            text=[f"<${low_income_threshold:,.0f}", 
                                  f"${low_income_threshold:,.0f}-${high_income_threshold:,.0f}",
                                  f">${high_income_threshold:,.0f}"],
                            textposition='outside'
                        )
                    ])
                    strat_fig.update_layout(
                        title="<b>Income Stratification</b>",
                        xaxis_title="Income Category",
                        yaxis_title="Number of Stations",
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(strat_fig, use_container_width=True)
                    
                    # Income inequality metric
                    if len(income_data) > 1:
                        gini_approx = (income_data.std() / income_data.mean()) * 0.5  # Simplified Gini approximation
                        st.metric("Income Inequality (Gini approx.)", f"{gini_approx:.3f}",
                                 delta="Higher = more inequality")
        
        # Economic Indicators Summary Table
        if 'median_household_income' in stations_df.columns or 'poverty_rate' in stations_df.columns:
            st.markdown("### Economic Indicators Summary")
            
            econ_summary = []
            
            if 'median_household_income' in stations_df.columns:
                income = stations_df['median_household_income'].dropna()
                if len(income) > 0:
                    econ_summary.append({
                        'Indicator': 'Household Income',
                        'Median': f"${income.median():,.0f}",
                        'Mean': f"${income.mean():,.0f}",
                        'Std Dev': f"${income.std():,.0f}",
                        'Min': f"${income.min():,.0f}",
                        'Max': f"${income.max():,.0f}"
                    })
            
            if 'poverty_rate' in stations_df.columns:
                poverty = stations_df['poverty_rate'].dropna()
                if len(poverty) > 0:
                    econ_summary.append({
                        'Indicator': 'Poverty Rate',
                        'Median': f"{poverty.median():.1f}%",
                        'Mean': f"{poverty.mean():.1f}%",
                        'Std Dev': f"{poverty.std():.1f}%",
                        'Min': f"{poverty.min():.1f}%",
                        'Max': f"{poverty.max():.1f}%"
                    })
            
            if 'population_density' in stations_df.columns:
                density = stations_df['population_density'].dropna()
                if len(density) > 0:
                    econ_summary.append({
                        'Indicator': 'Population Density',
                        'Median': f"{density.median():,.0f}/km²",
                        'Mean': f"{density.mean():,.0f}/km²",
                        'Std Dev': f"{density.std():,.0f}/km²",
                        'Min': f"{density.min():,.0f}/km²",
                        'Max': f"{density.max():,.0f}/km²"
                    })
            
            if 'percent_college_educated' in stations_df.columns:
                college = stations_df['percent_college_educated'].dropna()
                if len(college) > 0:
                    econ_summary.append({
                        'Indicator': 'College Educated',
                        'Median': f"{college.median():.1f}%",
                        'Mean': f"{college.mean():.1f}%",
                        'Std Dev': f"{college.std():.1f}%",
                        'Min': f"{college.min():.1f}%",
                        'Max': f"{college.max():.1f}%"
                    })
            
            if 'unemployment_rate' in stations_df.columns:
                unemployment = stations_df['unemployment_rate'].dropna()
                if len(unemployment) > 0:
                    econ_summary.append({
                        'Indicator': 'Unemployment Rate',
                        'Median': f"{unemployment.median():.1f}%",
                        'Mean': f"{unemployment.mean():.1f}%",
                        'Std Dev': f"{unemployment.std():.1f}%",
                        'Min': f"{unemployment.min():.1f}%",
                        'Max': f"{unemployment.max():.1f}%"
                    })
            
            if 'median_home_value' in stations_df.columns:
                home_value = stations_df['median_home_value'].dropna()
                if len(home_value) > 0:
                    econ_summary.append({
                        'Indicator': 'Median Home Value',
                        'Median': f"${home_value.median():,.0f}",
                        'Mean': f"${home_value.mean():,.0f}",
                        'Std Dev': f"${home_value.std():,.0f}",
                        'Min': f"${home_value.min():,.0f}",
                        'Max': f"${home_value.max():,.0f}"
                    })
            
            if 'per_capita_income' in stations_df.columns:
                per_capita = stations_df['per_capita_income'].dropna()
                if len(per_capita) > 0:
                    econ_summary.append({
                        'Indicator': 'Per Capita Income',
                        'Median': f"${per_capita.median():,.0f}",
                        'Mean': f"${per_capita.mean():,.0f}",
                        'Std Dev': f"${per_capita.std():,.0f}",
                        'Min': f"${per_capita.min():,.0f}",
                        'Max': f"${per_capita.max():,.0f}"
                    })
            
            if econ_summary:
                econ_df = pd.DataFrame(econ_summary)
                st.dataframe(econ_df, use_container_width=True, hide_index=True)
                
                # Additional economic insights
                if 'median_household_income' in stations_df.columns and 'mean_household_income' in stations_df.columns:
                    st.markdown("**Income Analysis:**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        median_income = stations_df['median_household_income'].median()
                        mean_income = stations_df['mean_household_income'].median()
                        income_gap = mean_income - median_income
                        st.metric("Income Gap (Mean - Median)", f"${income_gap:,.0f}",
                                 delta="Higher = more inequality")
                    
                    with col2:
                        if 'income_inequality_ratio' in stations_df.columns:
                            avg_inequality = stations_df['income_inequality_ratio'].mean()
                            st.metric("Income Inequality Ratio", f"{avg_inequality:.2f}",
                                     delta="Mean/Median ratio")
                    
                    with col3:
                        if 'economic_diversity_index' in stations_df.columns:
                            avg_diversity = stations_df['economic_diversity_index'].mean()
                            st.metric("Economic Diversity Index", f"{avg_diversity:.2f}",
                                     delta="Higher = more stable")
        
        # Economic vs Water Stress Analysis
        if 'WSLUS' in stations_df.columns and 'median_household_income' in stations_df.columns:
            st.markdown("### Economic Factors vs Water Stress")
            
            # Calculate correlations
            econ_correlations = {}
            if 'median_household_income' in stations_df.columns:
                corr = stations_df['median_household_income'].corr(stations_df['WSLUS'])
                if not pd.isna(corr):
                    econ_correlations['Household Income'] = corr
            
            if 'poverty_rate' in stations_df.columns:
                corr = stations_df['poverty_rate'].corr(stations_df['WSLUS'])
                if not pd.isna(corr):
                    econ_correlations['Poverty Rate'] = corr
            
            if 'population_density' in stations_df.columns:
                corr = stations_df['population_density'].corr(stations_df['WSLUS'])
                if not pd.isna(corr):
                    econ_correlations['Population Density'] = corr
            
            if 'percent_college_educated' in stations_df.columns:
                corr = stations_df['percent_college_educated'].corr(stations_df['WSLUS'])
                if not pd.isna(corr):
                    econ_correlations['College Education'] = corr
            
            if 'unemployment_rate' in stations_df.columns:
                corr = stations_df['unemployment_rate'].corr(stations_df['WSLUS'])
                if not pd.isna(corr):
                    econ_correlations['Unemployment Rate'] = corr
            
            if 'median_home_value' in stations_df.columns:
                corr = stations_df['median_home_value'].corr(stations_df['WSLUS'])
                if not pd.isna(corr):
                    econ_correlations['Home Value'] = corr
            
            if 'per_capita_income' in stations_df.columns:
                corr = stations_df['per_capita_income'].corr(stations_df['WSLUS'])
                if not pd.isna(corr):
                    econ_correlations['Per Capita Income'] = corr
            
            if econ_correlations:
                corr_df = pd.DataFrame({
                    'Economic Indicator': list(econ_correlations.keys()),
                    'Correlation with WSLUS': [f"{v:.3f}" for v in econ_correlations.values()],
                    'Strength': [abs(v) for v in econ_correlations.values()]
                }).sort_values('Strength', ascending=False)
                
                st.markdown("**Correlation Analysis:**")
                st.dataframe(corr_df[['Economic Indicator', 'Correlation with WSLUS']], 
                           use_container_width=True, hide_index=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Scatter plot: Income vs WSLUS
                income_fig = go.Figure()
                
                # Color by EJ status if available
                if 'in_ej_community' in stations_df.columns:
                    ej_stations = stations_df[stations_df['in_ej_community'] == 1]
                    non_ej_stations = stations_df[stations_df['in_ej_community'] == 0]
                    
                    income_fig.add_trace(go.Scatter(
                        x=non_ej_stations['median_household_income'],
                        y=non_ej_stations['WSLUS'],
                        mode='markers',
                        name='Non-EJ Areas',
                        marker=dict(color='#3b82f6', size=8, opacity=0.6)
                    ))
                    
                    income_fig.add_trace(go.Scatter(
                        x=ej_stations['median_household_income'],
                        y=ej_stations['WSLUS'],
                        mode='markers',
                        name='EJ Communities',
                        marker=dict(color='#ef4444', size=10, opacity=0.8, symbol='diamond')
                    ))
                else:
                    income_fig.add_trace(go.Scatter(
                        x=stations_df['median_household_income'],
                        y=stations_df['WSLUS'],
                        mode='markers',
                        name='All Stations',
                        marker=dict(color='#3b82f6', size=8, opacity=0.6)
                    ))
                
                income_fig.update_layout(
                    title="<b>Household Income vs Water Stress</b>",
                    xaxis_title="Median Household Income ($)",
                    yaxis_title="WSLUS Score",
                    height=400,
                    hovermode='closest'
                )
                st.plotly_chart(income_fig, use_container_width=True)
                
                # Income-based analysis
                if len(income_data) > 0:
                    # Compare high vs low income areas
                    low_income_threshold = income_data.quantile(0.33)
                    high_income_threshold = income_data.quantile(0.67)
                    
                    low_income_stations = stations_df[stations_df['median_household_income'] < low_income_threshold]
                    high_income_stations = stations_df[stations_df['median_household_income'] >= high_income_threshold]
                    
                    if len(low_income_stations) > 0 and len(high_income_stations) > 0:
                        low_income_wslus = low_income_stations['WSLUS'].mean()
                        high_income_wslus = high_income_stations['WSLUS'].mean()
                        
                        if low_income_wslus > high_income_wslus:
                            st.warning(f"**Income Disparity**: Low-income areas (<${low_income_threshold:,.0f}) have {low_income_wslus - high_income_wslus:.1f} points higher water stress than high-income areas (>{high_income_threshold:,.0f}).")
                
                # EJ insight
                if 'in_ej_community' in stations_df.columns:
                    ej_stations_subset = stations_df[stations_df['in_ej_community'] == 1]
                    non_ej_stations_subset = stations_df[stations_df['in_ej_community'] == 0]
                    
                    if len(ej_stations_subset) > 0 and len(non_ej_stations_subset) > 0:
                        ej_wslus = ej_stations_subset['WSLUS'].mean()
                        non_ej_wslus = non_ej_stations_subset['WSLUS'].mean()
                        if ej_wslus > non_ej_wslus:
                            st.warning(f"**Environmental Justice Gap**: EJ communities have {ej_wslus - non_ej_wslus:.1f} points higher water stress on average. Priority intervention needed.")
            
            with col2:
                # Poverty vs WSLUS
                if 'poverty_rate' in stations_df.columns:
                    poverty_fig = go.Figure()
                    
                    if 'in_ej_community' in stations_df.columns:
                        ej_stations = stations_df[stations_df['in_ej_community'] == 1]
                        non_ej_stations = stations_df[stations_df['in_ej_community'] == 0]
                        
                        poverty_fig.add_trace(go.Scatter(
                            x=non_ej_stations['poverty_rate'],
                            y=non_ej_stations['WSLUS'],
                            mode='markers',
                            name='Non-EJ Areas',
                            marker=dict(color='#10b981', size=8, opacity=0.6)
                        ))
                        
                        poverty_fig.add_trace(go.Scatter(
                            x=ej_stations['poverty_rate'],
                            y=ej_stations['WSLUS'],
                            mode='markers',
                            name='EJ Communities',
                            marker=dict(color='#ef4444', size=10, opacity=0.8, symbol='diamond')
                        ))
                    else:
                        poverty_fig.add_trace(go.Scatter(
                            x=stations_df['poverty_rate'],
                            y=stations_df['WSLUS'],
                            mode='markers',
                            name='All Stations',
                            marker=dict(color='#10b981', size=8, opacity=0.6)
                        ))
                    
                    poverty_fig.update_layout(
                        title="<b>Poverty Rate vs Water Stress</b>",
                        xaxis_title="Poverty Rate (%)",
                        yaxis_title="WSLUS Score",
                        height=400,
                        hovermode='closest'
                    )
                    st.plotly_chart(poverty_fig, use_container_width=True)
                    
                    # Poverty-based stratification
                    poverty_data_check = stations_df['poverty_rate'].dropna()
                    if len(poverty_data_check) > 0:
                        high_poverty_threshold = poverty_data_check.quantile(0.75)
                        low_poverty_threshold = poverty_data_check.quantile(0.25)
                        
                        high_poverty_stations = stations_df[stations_df['poverty_rate'] > high_poverty_threshold]
                        low_poverty_stations = stations_df[stations_df['poverty_rate'] < low_poverty_threshold]
                        
                        if len(high_poverty_stations) > 0 and len(low_poverty_stations) > 0:
                            high_pov_wslus = high_poverty_stations['WSLUS'].mean()
                            low_pov_wslus = low_poverty_stations['WSLUS'].mean()
                            
                            if high_pov_wslus > low_pov_wslus:
                                st.warning(f"**Poverty Impact**: High-poverty areas (>={high_poverty_threshold:.1f}%) have {high_pov_wslus - low_pov_wslus:.1f} points higher water stress than low-poverty areas (<{low_poverty_threshold:.1f}%).")
                    
                    # Correlation insight
                    corr = stations_df['poverty_rate'].corr(stations_df['WSLUS'])
                    if abs(corr) > 0.3:
                        direction = "increases" if corr > 0 else "decreases"
                        strength = "strong" if abs(corr) > 0.6 else "moderate"
                        st.info(f"**Correlation**: {strength.capitalize()} {direction} correlation between poverty rate and water stress (r={corr:.2f})")
    else:
        st.info(" **Economic data**: Run the notebook with Census data integration to see economic indicators")
    
    st.markdown("---")
    
    # ========== VISUALIZATION 3: Stress Drivers ==========
    st.markdown("## What's Driving Water Stress?")
    
    if 'conductivity_stress' in stations_df.columns:
        stress_cols = ['conductivity_stress', 'tds_stress', 'ph_stress', 
                      'temperature_stress', 'do_stress']
        available_stress = [c for c in stress_cols if c in stations_df.columns]
        
        if available_stress:
            stress_means = stations_df[available_stress].mean().sort_values(ascending=False)
            
            # Clean up labels for display
            clean_labels = {
                'conductivity_stress': 'Road Salt / Runoff',
                'tds_stress': 'Dissolved Solids',
                'ph_stress': 'pH Imbalance',
                'temperature_stress': 'Temperature',
                'do_stress': 'Dissolved Oxygen'
            }
            
            stress_labels = [clean_labels.get(col, col.replace('_', ' ').title()) for col in stress_means.index]
            
            # Horizontal bar chart
            stress_fig = go.Figure(data=[
                go.Bar(
                    y=stress_labels,
                    x=stress_means.values * 100,  # Convert to percentage
                    orientation='h',
                    marker=dict(
                        color=stress_means.values * 100,
                        colorscale='Reds',
                        showscale=True,
                        colorbar=dict(title="Stress %")
                    ),
                    text=[f"{v*100:.1f}%" for v in stress_means.values],
                    textposition='outside'
                )
            ])
            stress_fig.update_layout(
                title="<b>Primary Stress Drivers (Average Impact)</b>",
                xaxis_title="Average Stress Level (%)",
                yaxis_title="",
                height=400,
                showlegend=False
            )
            st.plotly_chart(stress_fig, use_container_width=True)
            
            # Insight text
            top_driver = stress_labels[0]
            top_value = stress_means.values[0] * 100
            st.info(f"**Key Insight**: {top_driver} is the primary stress driver, contributing {top_value:.1f}% on average. Focus conservation efforts on addressing this factor first.")
    
    st.markdown("---")
    
    # ========== VISUALIZATION 5: Preservation Opportunities ==========
    st.markdown("## Top Preservation Opportunities")
    
    try:
        opp_df = pd.read_csv('preservation_opportunities.csv')
        
        if len(opp_df) > 0:
            # Summary metrics with business insights
            col1, col2, col3 = st.columns(3)
            
            total_area = opp_df['area_hectares'].sum()
            total_impact = opp_df['estimated_impact'].sum()
            top_area = opp_df['area_hectares'].max()
            
            # Calculate business-focused metrics
            # Rough cost estimate: $50k per hectare (conservation land acquisition average)
            estimated_cost = total_area * 50000
            cost_per_impact = estimated_cost / total_impact if total_impact > 0 else 0
            
            # Calculate potential water quality improvement
            # Estimate: preserving 1 hectare reduces WSLUS by ~0.1-0.3 points on average
            avg_wslus_reduction = total_area * 0.2  # Conservative estimate
            
            # Calculate area comparison (football fields, Central Park, etc.)
            football_fields = total_area / 0.714  # 1 hectare ≈ 1.4 football fields
            central_park_equiv = total_area / 341  # Central Park is 341 hectares
            
            with col1:
                st.metric(
                    "Total Conservation Investment", 
                    f"{total_area:.0f} hectares",
                    delta=f"~${estimated_cost/1e6:.1f}M estimated cost"
                )
                st.caption(f"{len(opp_df)} priority parcels | ~{football_fields:.0f} football fields")
            
            with col2:
                st.metric(
                    "Potential Water Quality Improvement", 
                    f"{avg_wslus_reduction:.1f} points",
                    delta=f"WSLUS reduction potential"
                )
                st.caption(f"Impact score: {total_impact:,.0f} | ${cost_per_impact:,.0f} per impact point")
            
            with col3:
                st.metric(
                    "Highest Priority Target", 
                    f"{top_area:.1f} hectares",
                    delta=f"Top-ranked opportunity"
                )
                top_opp = opp_df.loc[opp_df['area_hectares'].idxmax()]
                top_impact = top_opp.get('estimated_impact', 0)
                st.caption(f"Impact: {top_impact:,.0f} | {top_opp.get('landuse_type', 'Unknown')} land")
            
            # Business Value Summary
            st.markdown("### Strategic Value & ROI")
            
            # Calculate metrics for business insights
            avg_impact_per_ha = total_impact / total_area if total_area > 0 else 0
            top_10_impact = opp_df.head(10)['estimated_impact'].sum()
            top_10_pct = (top_10_impact / total_impact * 100) if total_impact > 0 else 0
            top_10_area = opp_df.head(10)['area_hectares'].sum()
            
            # EJ priority percentage
            ej_pct = 0
            ej_opps_count = 0
            if 'ej_priority' in opp_df.columns:
                ej_opps = opp_df[opp_df['ej_priority'] == 1]
                ej_opps_count = len(ej_opps)
                ej_pct = ej_opps_count / len(opp_df) * 100 if len(opp_df) > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Cost efficiency
                if cost_per_impact > 0:
                    st.metric(
                        "Cost Efficiency",
                        f"${cost_per_impact:,.0f}",
                        delta="Per impact point"
                    )
                    st.caption("Lower = better ROI")
            
            with col2:
                # Average impact per hectare
                st.metric(
                    "Impact per Hectare",
                    f"{avg_impact_per_ha:,.0f}",
                    delta="Average efficiency"
                )
                st.caption("Higher = more value per area")
            
            with col3:
                # Top 10 concentration
                st.metric(
                    "Top 10 Concentration",
                    f"{top_10_pct:.1f}%",
                    delta="Of total impact"
                )
                st.caption(f"Focus on top 10 ({top_10_area:.0f} ha) for 80/20 rule")
            
            with col4:
                # EJ priority percentage
                if ej_pct > 0:
                    st.metric(
                        "EJ Priority Opportunities",
                        f"{ej_pct:.1f}%",
                        delta=f"{ej_opps_count} parcels"
                    )
                    st.caption("Environmental justice focus")
                elif 'wslus_score' in opp_df.columns:
                    high_stress = len(opp_df[opp_df['wslus_score'] > 50])
                    st.metric(
                        "High Stress Areas",
                        f"{high_stress}",
                        delta=f"{high_stress/len(opp_df)*100:.1f}%"
                    )
                    st.caption("WSLUS > 50")
                else:
                    st.metric(
                        "Total Opportunities",
                        f"{len(opp_df)}",
                        delta="Priority parcels"
                    )
                    st.caption("Ranked by impact")
            
            # Business Insights & Recommendations
            st.markdown("### Business Insights & Strategic Recommendations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Investment Summary:**
                - **Total Investment Required**: ~${:.1f}M for {:.0f} hectares
                - **Expected Water Quality Improvement**: {:.1f} point WSLUS reduction
                - **Cost per Impact Point**: ${:,.0f}
                - **Strategic Focus**: Top 10 opportunities represent {:.1f}% of total impact
                
                **Key Insight**: Focusing on the top 10 opportunities provides {:.1f}% of the total impact 
                with significantly less area and cost, following the 80/20 principle.
                """.format(
                    estimated_cost/1e6,
                    total_area,
                    avg_wslus_reduction,
                    cost_per_impact,
                    top_10_pct,
                    top_10_pct
                ))
            
            with col2:
                remaining_area = total_area - top_10_area
                num_stations = opp_df['station_id'].nunique() if 'station_id' in opp_df.columns else len(opp_df)
                
                st.markdown("""
                **Strategic Recommendations:**
                
                1. **Phase 1 (Immediate)**: Prioritize top 10 opportunities
                   - Focus on highest impact-to-cost ratio
                   - Target: {:.0f} hectares for maximum ROI
                   - Expected impact: {:.1f}% of total benefit
                
                2. **Phase 2 (Short-term)**: Address EJ priority areas
                   - {:.1f}% of opportunities are in EJ communities ({:.0f} parcels)
                   - Dual benefit: Environmental + Social equity
                   - Aligns with state EJ priorities
                
                3. **Phase 3 (Long-term)**: Complete portfolio
                   - Remaining {:.0f} hectares for comprehensive protection
                   - Builds on Phase 1 & 2 success
                   - Ensures watershed-wide coverage
                
                **Expected Outcomes:**
                - Improved water quality across {:.0f} monitoring stations
                - Protection of critical watershed areas
                - Enhanced environmental justice outcomes
                - Long-term ecosystem resilience
                """.format(
                    top_10_area,
                    top_10_pct,
                    ej_pct,
                    ej_opps_count,
                    remaining_area,
                    num_stations
                ))
            
            # Top opportunities visualization
            top_20 = opp_df.head(20)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Horizontal bar chart of top opportunities
                opp_fig = go.Figure(data=[
                    go.Bar(
                        y=[f"Opportunity #{i}" for i in top_20['rank']],
                        x=top_20['estimated_impact'],
                        orientation='h',
                        marker=dict(
                            color=top_20['estimated_impact'],
                            colorscale='Viridis',
                            showscale=True
                        ),
                        text=[f"{row['area_hectares']:.1f} ha" for _, row in top_20.iterrows()],
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Impact: %{x:,.0f}<br>Area: %{text}<extra></extra>'
                    )
                ])
                opp_fig.update_layout(
                    title="<b>Top 20 Preservation Opportunities (Ranked by Impact)</b>",
                    xaxis_title="Estimated Impact Score",
                    yaxis_title="",
                    height=600,
                    showlegend=False
                )
                st.plotly_chart(opp_fig, use_container_width=True)
            
            with col2:
                st.markdown("### Opportunity Details")
                
                # Land use breakdown
                if 'landuse_type' in opp_df.columns:
                    landuse_counts = opp_df['landuse_type'].value_counts()
                    
                    landuse_pie = go.Figure(data=[
                        go.Pie(
                            labels=landuse_counts.index,
                            values=landuse_counts.values,
                            hole=0.3,
                            marker=dict(colors=['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6'])
                        )
                    ])
                    landuse_pie.update_layout(
                        title="<b>By Land Type</b>",
                        height=300,
                        showlegend=True
                    )
                    st.plotly_chart(landuse_pie, use_container_width=True)
                    
                    st.markdown("**Breakdown:**")
                    for landuse, count in landuse_counts.items():
                        area = opp_df[opp_df['landuse_type'] == landuse]['area_hectares'].sum()
                        st.write(f"- **{landuse}**: {count} parcels ({area:.1f} ha)")
            
            # Explanation of Estimated Impact
            st.markdown("### What Does Estimated Impact Mean?")
            
            with st.expander("Understanding Estimated Impact", expanded=False):
                st.markdown("""
                **Estimated Impact** is a ranking metric that helps prioritize conservation opportunities by combining:
                
                **Formula**: `Estimated Impact = Preservation Score × Area (hectares)`
                
                **What It Measures:**
                - **Total Conservation Benefit**: Combines the quality/priority of an area with its size
                - **Relative Ranking**: Higher impact = higher priority for conservation action
                - **Resource Efficiency**: Helps maximize conservation benefit per dollar invested
                
                **Key Points:**
                1. **Not an Absolute Measure**: Impact scores are relative - use them to compare opportunities, not as absolute predictions
                2. **Quality × Scale**: A small high-priority area (Score=90, Area=2ha → Impact=180) vs. a large moderate area (Score=45, Area=5ha → Impact=225)
                3. **Prioritization Tool**: The opportunity with the highest estimated impact should be addressed first
                
                **Real-World Meaning:**
                - Higher impact opportunities are likely to provide greater water quality improvements
                - Larger areas typically provide more ecosystem services (filtration, habitat, flood control)
                - Combining quality and size helps balance "quick wins" (high score, small area) with "big wins" (moderate score, large area)
                
                **Example Interpretation:**
                - Impact = 500: High priority, large area, or both → Should be prioritized
                - Impact = 200: Moderate priority → Consider if resources allow
                - Impact = 50: Lower priority → May be deferred unless other factors (e.g., EJ status) require action
                """)
            
            # Detailed opportunity information table
            st.markdown("### Detailed Opportunity Information")
            
            # Create detailed table with all available information
            display_cols = ['rank', 'station_id', 'landuse_type', 'area_hectares', 
                           'preservation_score', 'estimated_impact', 'wslus_score']
            available_cols = [col for col in display_cols if col in opp_df.columns]
            
            if available_cols:
                # Format the dataframe for display
                display_df = opp_df[available_cols].head(20).copy()
                
                # Format numeric columns
                if 'area_hectares' in display_df.columns:
                    display_df['area_hectares'] = display_df['area_hectares'].apply(lambda x: f"{x:.1f} ha")
                if 'preservation_score' in display_df.columns:
                    display_df['preservation_score'] = display_df['preservation_score'].apply(lambda x: f"{x:.1f}")
                if 'estimated_impact' in display_df.columns:
                    display_df['estimated_impact'] = display_df['estimated_impact'].apply(lambda x: f"{x:,.0f}")
                if 'wslus_score' in display_df.columns:
                    display_df['wslus_score'] = display_df['wslus_score'].apply(lambda x: f"{x:.1f}")
                
                # Rename columns for display
                display_df.columns = [col.replace('_', ' ').title() for col in display_df.columns]
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Information about how opportunities are identified
            with st.expander("How Preservation Opportunities Are Identified", expanded=False):
                st.markdown("""
                ### Selection Criteria
                
                **Primary Factors:**
                1. **Water Stress Level**: Stations with WSLUS scores above the 75th percentile (or 40, whichever is lower)
                2. **Environmental Justice Priority**: Areas in EJ communities receive higher priority
                3. **Economic Vulnerability**: Low-income areas with high water stress are prioritized
                4. **Spatial Proximity**: Areas within 2km upstream of high-stress monitoring stations
                
                ### Scoring Methodology
                
                **Preservation Score** combines:
                - WSLUS Score (60% weight): Higher water stress = higher priority
                - EJ Community Flag (30% weight): EJ areas get priority boost
                - Area Size (10% weight): Larger areas have greater potential impact
                
                **Estimated Impact** = Preservation Score × Area (hectares)
                
                **What Estimated Impact Represents:**
                
                Estimated Impact is a **composite measure of total conservation benefit** that combines:
                1. **Preservation Score** (quality/priority of the area)
                2. **Area in hectares** (size/scale of the opportunity)
                
                **Interpretation:**
                - **Higher Impact = Higher Priority**: Opportunities with higher estimated impact should be prioritized for conservation
                - **Combines Quality and Scale**: A small high-priority area might have similar impact to a larger moderate-priority area
                - **Resource Allocation**: Helps allocate limited conservation resources to maximize total benefit
                
                **Example:**
                - Opportunity A: Preservation Score = 80, Area = 5 ha → Estimated Impact = 400
                - Opportunity B: Preservation Score = 50, Area = 10 ha → Estimated Impact = 500
                - **Opportunity B has higher impact** (500 vs 400) despite lower preservation score, because its larger size provides more total benefit
                
                **Why Multiply by Area?**
                - Larger preserved areas provide more ecosystem services (filtration, habitat, carbon storage)
                - Bigger areas have greater potential to reduce water stress across the watershed
                - Conservation investments scale with area, so impact should reflect size
                
                **Limitations:**
                - Impact is a **relative ranking tool**, not an absolute measure of WSLUS reduction
                - Actual water quality improvements would require field validation
                - Does not account for implementation costs or feasibility
                
                ### Data Sources
                - **Water Quality**: Monitoring station data (pH, conductivity, TDS, temperature, DO)
                - **Economic Data**: MA DOR municipal income data
                - **Location Data**: Station coordinates and location descriptions
                
                ### Limitations
                - Opportunities are based on station characteristics when GIS land use data is unavailable
                - Actual parcel boundaries would require MassGIS land use shapefiles
                - Impact estimates are projections based on water quality and economic factors
                """)
        else:
            st.info("No preservation opportunities found. Run the full analysis to identify conservation targets.")
    except Exception as e:
        st.info(" **Preservation opportunities**: Run the notebook analysis to generate ranked conservation targets.")
    
    st.markdown("---")
    
    # ========== HOW WE CALCULATE INTERVENTION PRIORITY ==========
    st.markdown("## How We Calculate Intervention Priority")
    
    with st.expander("Intervention Priority Methodology", expanded=True):
        st.markdown("""
        ### Purpose
        
        Intervention Priority combines **water stress** and **economic vulnerability** to identify areas where conservation efforts will have the greatest impact on both environmental and social outcomes.
        
        ### Calculation Steps
        
        **Step 1: Normalize Water Stress (WSLUS)**
        - Convert WSLUS scores to 0-1 scale
        - Formula: `(WSLUS - min(WSLUS)) / (max(WSLUS) - min(WSLUS))`
        - Ensures all stations are on the same scale regardless of absolute values
        
        **Step 2: Normalize Economic Vulnerability**
        - Uses one of the following (in priority order):
          1. `ej_vulnerability` score (if available) - already 0-1 scale
          2. `ej_vulnerability_proxy` - calculated from income thresholds
          3. `poverty_rate` - normalized to 0-1 scale
        - Higher values = more economically vulnerable
        
        **Step 3: Calculate Intervention Priority**
        - Weighted combination: `(WSLUS_normalized × 0.6) + (EJ_vulnerability × 0.4)`
        - **60% weight on water stress**: Environmental impact is primary
        - **40% weight on economic vulnerability**: Social equity is secondary
        - Result: 0-1 scale (higher = higher priority)
        
        **Step 4: Categorize Priority Levels**
        - **Low (0-0.4)**: Monitor, low intervention priority
        - **Medium (0.4-0.6)**: Moderate priority, consider intervention
        - **High (0.6-0.8)**: High priority, intervention recommended
        - **Critical (0.8-1.0)**: Immediate intervention required
        
        ### Why This Formula?
        
        - **Balanced**: Considers both environmental and social factors
        - **Prioritized**: Water stress gets higher weight (environmental impact is primary)
        - **Equitable**: Economic vulnerability ensures EJ communities are prioritized
        - **Actionable**: Clear categories guide resource allocation
        
        ### Example Calculation
        
        Station A:
        - WSLUS = 70 (normalized: 0.85)
        - EJ vulnerability = 0.8
        - Intervention Priority = (0.85 × 0.6) + (0.8 × 0.4) = **0.83** → **High Priority**
        
        Station B:
        - WSLUS = 30 (normalized: 0.30)
        - EJ vulnerability = 0.2
        - Intervention Priority = (0.30 × 0.6) + (0.2 × 0.4) = **0.26** → **Low Priority**
        """)
    
    st.markdown("---")
    st.markdown("## Intervention Priority Results")
    
    if 'WSLUS' in stations_df.columns:
        # Create intervention priority score
        intervention_df = stations_df.copy()
        
        # Normalize WSLUS (0-1)
        if intervention_df['WSLUS'].max() > 0:
            intervention_df['wslus_normalized'] = (intervention_df['WSLUS'] - intervention_df['WSLUS'].min()) / (intervention_df['WSLUS'].max() - intervention_df['WSLUS'].min())
        else:
            intervention_df['wslus_normalized'] = 0
        
        # Add economic vulnerability component
        if 'ej_vulnerability' in intervention_df.columns:
            intervention_df['ej_normalized'] = intervention_df['ej_vulnerability']
        elif 'ej_vulnerability_proxy' in intervention_df.columns:
            intervention_df['ej_normalized'] = intervention_df['ej_vulnerability_proxy']
        elif 'poverty_rate' in intervention_df.columns:
            # Use poverty rate as proxy
            intervention_df['ej_normalized'] = (intervention_df['poverty_rate'] - intervention_df['poverty_rate'].min()) / (intervention_df['poverty_rate'].max() - intervention_df['poverty_rate'].min() + 1e-6)
        else:
            intervention_df['ej_normalized'] = 0
        
        # Calculate intervention priority (weighted combination)
        intervention_df['intervention_priority'] = (
            intervention_df['wslus_normalized'] * 0.6 +  # Water stress is primary
            intervention_df['ej_normalized'] * 0.4      # Economic vulnerability is secondary
        )
        
        # Categorize priorities
        intervention_df['priority_level'] = pd.cut(
            intervention_df['intervention_priority'],
            bins=[0, 0.4, 0.6, 0.8, 1.0],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        # Display priority distribution
        col1, col2, col3, col4 = st.columns(4)
        
        priority_counts = intervention_df['priority_level'].value_counts()
        
        with col1:
            critical = priority_counts.get('Critical', 0)
            st.metric("Critical Priority", critical)
            st.caption(f"{critical/len(intervention_df)*100:.1f}% of stations")
        
        with col2:
            high = priority_counts.get('High', 0)
            st.metric("High Priority", high)
            st.caption(f"{high/len(intervention_df)*100:.1f}% of stations")
        
        with col3:
            medium = priority_counts.get('Medium', 0)
            st.metric("Medium Priority", medium)
            st.caption(f"{medium/len(intervention_df)*100:.1f}% of stations")
        
        with col4:
            low = priority_counts.get('Low', 0)
            st.metric("Low Priority", low)
            st.caption(f"{low/len(intervention_df)*100:.1f}% of stations")
        
        # Top intervention targets
        st.markdown("### Top 20 Intervention Targets")
        
        top_targets = intervention_df.nlargest(20, 'intervention_priority')[
            ['station_id', 'WSLUS', 'intervention_priority', 'priority_level']
        ].copy()
        
        # Add economic context if available
        if 'median_household_income' in intervention_df.columns:
            top_targets = top_targets.merge(
                intervention_df[['station_id', 'median_household_income', 'poverty_rate', 'in_ej_community']],
                on='station_id',
                how='left'
            )
            display_cols = ['station_id', 'WSLUS', 'intervention_priority', 'priority_level', 
                          'median_household_income', 'poverty_rate', 'in_ej_community']
        else:
            display_cols = ['station_id', 'WSLUS', 'intervention_priority', 'priority_level']
        
        # Format for display
        top_targets_display = top_targets[display_cols].copy()
        top_targets_display['intervention_priority'] = top_targets_display['intervention_priority'].round(3)
        top_targets_display['WSLUS'] = top_targets_display['WSLUS'].round(1)
        
        if 'median_household_income' in top_targets_display.columns:
            top_targets_display['median_household_income'] = top_targets_display['median_household_income'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
        if 'poverty_rate' in top_targets_display.columns:
            top_targets_display['poverty_rate'] = top_targets_display['poverty_rate'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
        if 'in_ej_community' in top_targets_display.columns:
            top_targets_display['in_ej_community'] = top_targets_display['in_ej_community'].apply(lambda x: "Yes" if x == 1 else "No")
        
        st.dataframe(top_targets_display, use_container_width=True, hide_index=True)
        
        # Show calculation details for top targets
        st.markdown("### How Top Targets Were Selected")
        
        with st.expander("Calculation Details for Top 5 Targets", expanded=False):
            top_5 = intervention_df.nlargest(5, 'intervention_priority')
            for idx, row in top_5.iterrows():
                st.markdown(f"""
                **{row.get('station_id', 'Unknown')}** - Priority: {row['intervention_priority']:.3f} ({row['priority_level']})
                - WSLUS: {row['WSLUS']:.1f} (normalized: {row['wslus_normalized']:.3f})
                - Economic Vulnerability: {row['ej_normalized']:.3f}
                - Calculation: ({row['wslus_normalized']:.3f} × 0.6) + ({row['ej_normalized']:.3f} × 0.4) = {row['intervention_priority']:.3f}
                """)
    
    st.markdown("---")
    
    # ========== RECOMMENDATIONS ==========
    st.markdown("## Recommended Actions")
    
    if 'WSLUS' in stations_df.columns:
        recommendations = []
        
        # High impervious stress
        if 'impervious_stress' in stations_df.columns:
            high_imperv = stations_df['impervious_stress'].mean()
            if high_imperv > 0.5:
                affected = len(stations_df[stations_df['impervious_stress'] > 0.5])
                recommendations.append({
                                        'title': 'Green Infrastructure Program',
                    'description': f'Install rain gardens, permeable pavements, and green roofs in {affected} high-stress areas.',
                    'impact': 'Reduce stormwater runoff and improve water quality',
                    'priority': 'High'
                })
        
        # EJ communities with economic context
        if 'in_ej_community' in stations_df.columns:
            ej_stations = stations_df[stations_df['in_ej_community'] == 1]
            if len(ej_stations) > 0:
                ej_avg = ej_stations['WSLUS'].mean() if 'WSLUS' in ej_stations.columns else 0
                
                # Add economic context
                ej_context = ""
                if 'median_household_income' in ej_stations.columns:
                    ej_income = ej_stations['median_household_income'].mean()
                    all_income = stations_df['median_household_income'].mean()
                    income_gap = all_income - ej_income
                    if income_gap > 10000:
                        ej_context = f"Average income ${ej_income:,.0f} (${income_gap:,.0f} below average)."
                
                if 'poverty_rate' in ej_stations.columns:
                    ej_poverty = ej_stations['poverty_rate'].mean()
                    all_poverty = stations_df['poverty_rate'].mean()
                    if ej_poverty > all_poverty * 1.2:
                        ej_context += f"Poverty rate {ej_poverty:.1f}% (above average)."
                
                recommendations.append({
                                        'title': 'Environmental Justice Priority',
                    'description': f'Focus conservation efforts on {len(ej_stations)} EJ communities with average stress of {ej_avg:.1f}.{ej_context}',
                    'impact': 'Address environmental inequities and protect vulnerable communities',
                    'priority': 'Critical'
                })
        
        # Economic-based interventions
        if 'median_household_income' in stations_df.columns and 'WSLUS' in stations_df.columns:
            # Find low-income, high-stress areas
            low_income_high_stress = stations_df[
                (stations_df['median_household_income'] < stations_df['median_household_income'].quantile(0.25)) &
                (stations_df['WSLUS'] > stations_df['WSLUS'].quantile(0.75))
            ]
            
            if len(low_income_high_stress) > 0:
                recommendations.append({
                                        'title': 'Economic & Environmental Intervention',
                    'description': f'{len(low_income_high_stress)} areas combine low income (<${low_income_high_stress["median_household_income"].max():,.0f}) with high water stress. Dual-benefit interventions recommended.',
                    'impact': 'Address both economic and environmental challenges simultaneously',
                    'priority': 'High'
                })
        
        # Road salt
        if 'conductivity_stress' in stations_df.columns:
            high_cond = stations_df['conductivity_stress'].mean()
            if high_cond > 0.6:
                affected = len(stations_df[stations_df['conductivity_stress'] > 0.6])
                recommendations.append({
                                        'title': 'Road Salt Reduction',
                    'description': f'Implement salt reduction strategies for {affected} affected areas.',
                    'impact': 'Reduce conductivity and improve aquatic ecosystem health',
                    'priority': 'High'
                })
        
        # Forest protection
        forest_cols = [c for c in stations_df.columns if 'forest' in c.lower() and '_pct' in c]
        if forest_cols:
            avg_forest = stations_df[forest_cols[0]].mean()
            if avg_forest < 30:
                recommendations.append({
                                        'title': 'Forest Protection Initiative',
                    'description': f'Current forest cover is {avg_forest:.1f}%. Increase protection to reduce water stress.',
                    'impact': 'Natural filtration and reduced runoff',
                    'priority': 'Medium'
                })
        
        # Display recommendations as cards
        if recommendations:
            for rec in recommendations:
                priority_labels = {
                    'Critical': 'CRITICAL',
                    'High': 'HIGH',
                    'Medium': 'MEDIUM',
                    'Low': 'LOW'
                }
                
                st.markdown(f"""
                <div style="border-left: 4px solid #3b82f6; padding: 20px; margin: 15px 0; background-color: #f8fafc; border-radius: 5px;">
                    <h3>{rec['title']} - {priority_labels.get(rec['priority'], rec['priority'].upper())} PRIORITY</h3>
                    <p style="font-size: 16px; color: #475569;">{rec['description']}</p>
                    <p style="font-size: 14px; color: #64748b; font-style: italic;">Impact: {rec['impact']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========== FINAL SUMMARY ==========
    st.markdown("## Summary & Next Steps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Key Findings")
        
        findings = []
        if 'WSLUS' in stations_df.columns:
            critical_pct = len(stations_df[stations_df['WSLUS'] > 75]) / len(stations_df) * 100
            if critical_pct > 10:
                findings.append(f"{critical_pct:.1f}% of monitoring stations are in critical condition")
            
            if 'in_ej_community' in stations_df.columns:
                ej_count = stations_df['in_ej_community'].sum()
                if ej_count > 0:
                    findings.append(f"{ej_count} Environmental Justice communities identified")
        
        if len(findings) == 0:
            findings.append("Analysis complete - review visualizations above for insights")
        
        for finding in findings:
            st.write(f"- {finding}")
    
    with col2:
        st.markdown("### Recommended Next Steps")
        st.markdown("""
        1. **Review Critical Areas** - Focus on stations with WSLUS > 75
        2. **Prioritize EJ Communities** - Address environmental justice concerns
        3. **Preserve High-Impact Parcels** - Use ranked opportunities list
        4. **Implement Green Infrastructure** - Reduce impervious surface impacts
        5. **Monitor Progress** - Track WSLUS improvements over time
        """)
    
    # Download options
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'WSLUS' in stations_df.columns:
            csv = stations_df.to_csv(index=False)
            st.download_button(
                label="Download Full Analysis (CSV)",
                data=csv,
                file_name=f"wslus_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        try:
            with open('executive_summary.json', 'r') as f:
                import json
                summary = json.load(f)
                summary_text = json.dumps(summary, indent=2)
                st.download_button(
                    label="Download Executive Summary (JSON)",
                    data=summary_text,
                    file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        except:
            st.info("Executive summary not available")

# Run the application
if __name__ == "__main__":
    main()