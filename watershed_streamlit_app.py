"""
Watershed Water Quality Analysis Dashboard
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
    page_icon="💧",
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
st.title("🌊 Watershed Water Quality Analysis Dashboard")
st.markdown("**Comprehensive analysis of water quality parameters, hotspots, and conservation priorities**")

# Sidebar for data upload and filters
with st.sidebar:
    st.header("📁 Data Configuration")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Water Quality Data (CSV)",
        type=['csv'],
        help="Upload your water quality dataset in CSV format"
    )
    
    # Use sample data option
    use_sample = st.checkbox("Use Sample Data", value=True if not uploaded_file else False)
    
    if uploaded_file is not None or use_sample:
        st.success("✅ Data loaded successfully!")
        
        st.markdown("---")
        st.header("⚙️ Analysis Settings")
        
        # Thresholds
        st.subheader("EPA Thresholds")
        do_threshold = st.slider("DO Minimum (mg/L)", 0.0, 10.0, 5.0, 0.5)
        ph_min = st.slider("pH Minimum", 5.0, 7.0, 6.5, 0.1)
        ph_max = st.slider("pH Maximum", 7.5, 10.0, 8.5, 0.1)
        conductivity_max = st.slider("Conductivity Max (µS/cm)", 200, 1000, 500, 50)
        tds_max = st.slider("TDS Max (mg/L)", 200, 1000, 500, 50)

@st.cache_data
def load_data(file=None):
    """Load and preprocess the water quality data."""
    if file is not None:
        df = pd.read_csv(file)
    else:
        # Generate sample data if no file is provided
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
        st.metric("Critical Priority Stations", critical_stations, 
                 delta=f"{critical_stations/len(hotspots)*100:.1f}% of total",
                 delta_color="inverse")
    
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
        ('Temperature_C_Numeric', 'Temperature (°C)', 1, 3, None, None),
        ('Specific_Conductivity_Numeric', 'Conductivity (µS/cm)', 2, 1, None, 500),
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
    
    # Load data
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    elif use_sample:
        df = load_data()
    else:
        st.warning("Please upload a CSV file or select 'Use Sample Data' to proceed.")
        return
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview", 
        "🌡️ Parameters", 
        "🌦️ Seasonal", 
        "💧 Stress Analysis",
        "📍 Hotspots",
        "📈 Dashboard",
        "🎯 WSLUS Model Insights"
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
            st.subheader("📊 Dataset Statistics")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.subheader("🏆 Top Priority Stations")
            top_priorities = hotspots.nlargest(10, 'hotspot_score')[
                ['Station_ID', 'Watershed', 'priority', 'hotspot_score']
            ]
            st.dataframe(top_priorities, use_container_width=True)
        
        # Timeline
        st.subheader("📅 Sampling Timeline")
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
        st.subheader("🌡️ Seasonal Statistics")
        
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
        st.subheader("🎯 Conservation Recommendations")
        
        critical_stations = hotspots[hotspots['priority'] == 'Critical']
        if len(critical_stations) > 0:
            st.warning(f"**{len(critical_stations)} stations require immediate intervention:**")
            
            for _, station in critical_stations.iterrows():
                st.write(f"• **{station['Station_ID']}** (Watershed: {station['Watershed']})")
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
            st.success("✅ Loaded WSLUS model results")
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
    """Create the WSLUS Model Insights tab with problem-solving approach"""
    st.header("🎯 Watershed Preservation Opportunity Analysis")
    st.markdown("**Using Water Stress from Land Use Score (WSLUS) Model**")
    
    # Try to load model results
    stations_df, model, feature_importance = load_wslus_model_results()
    
    if stations_df is None or len(stations_df) == 0:
        st.warning("""
        **WSLUS Model Results Not Found**
        
        To use this tab, please run the `watershed_preservation_model.ipynb` notebook first to generate:
        - `stations_with_wslus_scores.csv` - Station analysis with WSLUS scores
        - `wslus_predictive_model.pkl` - Trained model (optional)
        - `preservation_opportunities.csv` - Ranked preservation targets (optional)
        """)
        
        # Show sample/demo mode
        st.info("💡 **Demo Mode**: Showing example insights structure")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Critical Stations", "15", delta="12% of total")
        with col2:
            st.metric("High Priority Areas", "8", delta="Watersheds")
        with col3:
            st.metric("Preservation Opportunities", "42", delta="Parcels identified")
        
        st.markdown("---")
        st.subheader("🔍 Problem-Solving Framework")
        
        st.markdown("""
        ### Step 1: Identify Critical Areas
        - Stations with WSLUS > 75 (Critical risk)
        - EJ communities with high water stress
        - Areas with declining water quality trends
        
        ### Step 2: Understand Root Causes
        - High impervious surface coverage
        - Industrial land use nearby
        - Low forest/wetland protection
        - Stormwater runoff issues
        
        ### Step 3: Prioritize Interventions
        - Land preservation opportunities
        - Green infrastructure retrofits
        - Stormwater management improvements
        - Community engagement in EJ areas
        """)
        return
    
    # Convert to numeric for WSLUS if needed
    if 'WSLUS' in stations_df.columns:
        stations_df['WSLUS'] = pd.to_numeric(stations_df['WSLUS'], errors='coerce')
    
    # Problem-Solving Interface
    st.markdown("---")
    
    # Step 1: Problem Identification
    st.subheader("🔍 Step 1: Identify Critical Problems")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        critical = len(stations_df[stations_df['WSLUS'] > 75]) if 'WSLUS' in stations_df.columns else 0
        st.metric("Critical Risk Stations", critical, 
                 delta=f"{critical/len(stations_df)*100:.1f}%" if len(stations_df) > 0 else "0%",
                 delta_color="inverse")
    
    with col2:
        high_risk = len(stations_df[(stations_df['WSLUS'] > 50) & (stations_df['WSLUS'] <= 75)]) if 'WSLUS' in stations_df.columns else 0
        st.metric("High Risk Stations", high_risk,
                 delta=f"{high_risk/len(stations_df)*100:.1f}%" if len(stations_df) > 0 else "0%",
                 delta_color="off")
    
    with col3:
        ej_critical = len(stations_df[(stations_df.get('in_ej_community', 0) == 1) & 
                                     (stations_df.get('WSLUS', 0) > 75)]) if 'WSLUS' in stations_df.columns else 0
        st.metric("EJ Critical Areas", ej_critical,
                 delta="Requires attention",
                 delta_color="inverse")
    
    with col4:
        avg_wslus = stations_df['WSLUS'].mean() if 'WSLUS' in stations_df.columns else 0
        st.metric("Average WSLUS", f"{avg_wslus:.1f}",
                 delta="Target: <50",
                 delta_color="inverse" if avg_wslus > 50 else "normal")
    
    # Critical stations table
    if 'WSLUS' in stations_df.columns:
        critical_stations = stations_df[stations_df['WSLUS'] > 75].sort_values('WSLUS', ascending=False)
        
        if len(critical_stations) > 0:
            st.markdown("#### 🚨 Critical Priority Stations (WSLUS > 75)")
            
            display_cols = ['station_id', 'WSLUS', 'risk_category']
            if 'in_ej_community' in critical_stations.columns:
                display_cols.append('in_ej_community')
            if 'Latitude' in critical_stations.columns:
                display_cols.extend(['Latitude', 'Longitude'])
            
            available_cols = [c for c in display_cols if c in critical_stations.columns]
            st.dataframe(
                critical_stations[available_cols].head(20),
                use_container_width=True,
                hide_index=True
            )
    
    st.markdown("---")
    
    # Step 2: Root Cause Analysis
    st.subheader("🔬 Step 2: Understand Root Causes")
    
    if 'WSLUS' in stations_df.columns:
        # Feature importance if available
        if 'conductivity_stress' in stations_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Primary Stress Drivers")
                stress_cols = ['conductivity_stress', 'tds_stress', 'ph_stress', 
                              'temperature_stress', 'impervious_stress', 'landuse_stress']
                available_stress = [c for c in stress_cols if c in stations_df.columns]
                
                if available_stress:
                    stress_means = stations_df[available_stress].mean().sort_values(ascending=False)
                    stress_fig = go.Figure(data=[
                        go.Bar(x=stress_means.index, y=stress_means.values,
                              marker_color='coral')
                    ])
                    stress_fig.update_layout(
                        title="Average Stress by Component",
                        xaxis_title="Stress Component",
                        yaxis_title="Average Stress Score",
                        height=300
                    )
                    st.plotly_chart(stress_fig, use_container_width=True)
            
            with col2:
                st.markdown("##### Land Use Impact")
                # Show land use percentages if available
                landuse_cols = [c for c in stations_df.columns if any(x in c for x in ['forest', 'urban', 'industrial', 'wetland']) and '_pct' in c]
                
                if landuse_cols:
                    # Get most common buffer distance
                    buffer_dists = [500, 1000, 2000, 5000]
                    best_buffer = None
                    for dist in buffer_dists:
                        dist_cols = [c for c in landuse_cols if f'{dist}m' in c]
                        if dist_cols:
                            best_buffer = dist
                            break
                    
                    if best_buffer:
                        dist_cols = [c for c in landuse_cols if f'{best_buffer}m' in c]
                        landuse_means = stations_df[dist_cols].mean().sort_values(ascending=False)
                        
                        landuse_fig = go.Figure(data=[
                            go.Bar(x=landuse_means.index.str.replace(f'_{best_buffer}m_pct', ''),
                                  y=landuse_means.values,
                                  marker_color='lightgreen')
                        ])
                        landuse_fig.update_layout(
                            title=f"Average Land Use Coverage ({best_buffer}m buffer)",
                            xaxis_title="Land Use Type",
                            yaxis_title="Percentage Coverage",
                            height=300
                        )
                        st.plotly_chart(landuse_fig, use_container_width=True)
    
    st.markdown("---")
    
    # Step 3: Solutions & Opportunities
    st.subheader("💡 Step 3: Prioritized Solutions")
    
    # Try to load preservation opportunities
    try:
        opp_df = pd.read_csv('preservation_opportunities.csv')
        
        if len(opp_df) > 0:
            st.success(f"✅ Found {len(opp_df)} preservation opportunities")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Top 10 Preservation Opportunities")
                top_opps = opp_df.head(10)[['rank', 'landuse_type', 'area_hectares', 
                                           'preservation_score', 'estimated_impact']]
                st.dataframe(top_opps, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("##### Impact Summary")
                total_area = opp_df['area_hectares'].sum()
                total_impact = opp_df['estimated_impact'].sum()
                
                st.metric("Total Preservation Area", f"{total_area:.1f} hectares")
                st.metric("Total Impact Score", f"{total_impact:.0f}")
                st.metric("Priority Parcels", len(opp_df))
                
                # Land use breakdown
                if 'landuse_type' in opp_df.columns:
                    landuse_counts = opp_df['landuse_type'].value_counts()
                    st.markdown("**By Land Use Type:**")
                    for landuse, count in landuse_counts.items():
                        st.write(f"- {landuse}: {count} parcels")
        else:
            st.info("No preservation opportunities found. Run the full analysis in the notebook.")
    except:
        st.info("💡 **Preservation opportunities analysis**: Run the notebook to generate ranked preservation targets")
    
    # Action recommendations
    st.markdown("---")
    st.subheader("📋 Recommended Actions")
    
    if 'WSLUS' in stations_df.columns:
        # Generate recommendations based on data
        recommendations = []
        
        # Check for high impervious stress
        if 'impervious_stress' in stations_df.columns:
            high_imperv = stations_df['impervious_stress'].mean()
            if high_imperv > 0.5:
                recommendations.append({
                    'priority': 'High',
                    'action': 'Green Infrastructure Retrofits',
                    'description': f'High impervious coverage (avg stress: {high_imperv:.2f}). Install rain gardens, permeable pavements, and green roofs.',
                    'stations_affected': len(stations_df[stations_df['impervious_stress'] > 0.5])
                })
        
        # Check for EJ communities
        if 'in_ej_community' in stations_df.columns:
            ej_stations = stations_df[stations_df['in_ej_community'] == 1]
            if len(ej_stations) > 0:
                ej_avg_wslus = ej_stations['WSLUS'].mean() if 'WSLUS' in ej_stations.columns else 0
                recommendations.append({
                    'priority': 'Critical',
                    'action': 'EJ Community Water Quality Protection',
                    'description': f'{len(ej_stations)} EJ communities identified with average WSLUS of {ej_avg_wslus:.1f}. Prioritize these areas for intervention.',
                    'stations_affected': len(ej_stations)
                })
        
        # Check for high conductivity (road salt)
        if 'conductivity_stress' in stations_df.columns:
            high_cond = stations_df['conductivity_stress'].mean()
            if high_cond > 0.6:
                recommendations.append({
                    'priority': 'High',
                    'action': 'Road Salt Reduction Program',
                    'description': f'High conductivity stress ({high_cond:.2f}) indicates road salt runoff. Implement salt reduction strategies.',
                    'stations_affected': len(stations_df[stations_df['conductivity_stress'] > 0.6])
                })
        
        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            priority_color = {'Critical': '🔴', 'High': '🟠', 'Medium': '🟡', 'Low': '🟢'}.get(rec['priority'], '⚪')
            
            with st.expander(f"{priority_color} {rec['priority']} Priority: {rec['action']}"):
                st.write(rec['description'])
                st.metric("Stations Affected", rec['stations_affected'])
    
    # Model performance if available
    st.markdown("---")
    st.subheader("📊 Model Performance")
    
    try:
        with open('executive_summary.json', 'r') as f:
            import json
            summary = json.load(f)
            
            if 'overview' in summary:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model R² Score", 
                             f"{summary['overview'].get('model_accuracy', 0):.3f}" if summary['overview'].get('model_accuracy') else "N/A")
                with col2:
                    st.metric("Total Stations", summary['overview'].get('total_stations', 0))
                with col3:
                    st.metric("Mean WSLUS", f"{summary['overview'].get('mean_wslus', 0):.1f}")
    except:
        st.info("Model performance metrics not available. Run the full analysis to generate metrics.")

# Run the application
if __name__ == "__main__":
    main()