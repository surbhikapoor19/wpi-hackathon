#!/usr/bin/env python3
"""
MassGIS Data Acquisition and Processing Utilities
For Watershed Preservation Opportunity Map

This module provides functions to download and process GIS data from MassGIS
to support the Water Stress from Land Use Score (WSLUS) model.
"""

import os
import requests
import zipfile
import tempfile
from typing import Dict, List, Optional, Tuple
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
import rasterio
from rasterio.mask import mask
from rasterstats import zonal_stats
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MassGIS data URLs (update with actual URLs)
MASSGIS_URLS = {
    'impervious': 'https://s3.us-east-1.amazonaws.com/download.massgis.digital.mass.gov/shapefiles/state/impervious_surface_2016.zip',
    'landuse': 'https://s3.us-east-1.amazonaws.com/download.massgis.digital.mass.gov/shapefiles/state/landuse2016.zip',
    'watersheds': 'https://s3.us-east-1.amazonaws.com/download.massgis.digital.mass.gov/shapefiles/state/major_basins.zip',
    'ej_2020': 'https://s3.us-east-1.amazonaws.com/download.massgis.digital.mass.gov/shapefiles/state/ej_2020_census.zip',
    'protected_openspace': 'https://s3.us-east-1.amazonaws.com/download.massgis.digital.mass.gov/shapefiles/state/openspace.zip',
    'hydro': 'https://s3.us-east-1.amazonaws.com/download.massgis.digital.mass.gov/shapefiles/state/majhydro.zip',
    'wetlands': 'https://s3.us-east-1.amazonaws.com/download.massgis.digital.mass.gov/shapefiles/state/wetlandsdep.zip'
}

# Massachusetts State Plane Coordinate System
MA_CRS = 'EPSG:26986'  # NAD83 / Massachusetts Mainland


class MassGISDataLoader:
    """Handler for downloading and processing MassGIS data"""
    
    def __init__(self, cache_dir: str = './massgis_cache'):
        """
        Initialize the data loader
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def download_shapefile(self, data_type: str, force_download: bool = False) -> str:
        """
        Download and extract a shapefile from MassGIS
        
        Args:
            data_type: Type of data to download (key in MASSGIS_URLS)
            force_download: Force re-download even if cached
            
        Returns:
            Path to the extracted shapefile directory
        """
        if data_type not in MASSGIS_URLS:
            raise ValueError(f"Unknown data type: {data_type}")
            
        cache_path = os.path.join(self.cache_dir, data_type)
        
        if os.path.exists(cache_path) and not force_download:
            logger.info(f"Using cached {data_type} data")
            return cache_path
            
        logger.info(f"Downloading {data_type} from MassGIS...")
        url = MASSGIS_URLS[data_type]
        
        # Download the zip file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Save and extract
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
            
        os.makedirs(cache_path, exist_ok=True)
        
        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
            zip_ref.extractall(cache_path)
            
        os.unlink(tmp_path)
        logger.info(f"Successfully downloaded and extracted {data_type}")
        
        return cache_path
    
    def load_layer(self, data_type: str, target_crs: str = MA_CRS) -> gpd.GeoDataFrame:
        """
        Load a GIS layer and convert to target CRS
        
        Args:
            data_type: Type of data to load
            target_crs: Target coordinate reference system
            
        Returns:
            GeoDataFrame with the loaded data
        """
        data_path = self.download_shapefile(data_type)
        
        # Find the shapefile
        shapefiles = [f for f in os.listdir(data_path) if f.endswith('.shp')]
        if not shapefiles:
            raise FileNotFoundError(f"No shapefile found in {data_path}")
            
        shapefile_path = os.path.join(data_path, shapefiles[0])
        
        # Load and convert
        gdf = gpd.read_file(shapefile_path)
        if gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)
            
        logger.info(f"Loaded {data_type}: {len(gdf)} features")
        return gdf


def process_impervious_surface(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Process impervious surface data
    
    Args:
        gdf: Raw impervious surface GeoDataFrame
        
    Returns:
        Processed GeoDataFrame with standardized columns
    """
    processed = gdf.copy()
    
    # Standardize column names
    column_mapping = {
        'IMPERV_PCT': 'imperv_pct',
        'AREA_SQFT': 'area_sq_m',
        'YEAR': 'year'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in processed.columns:
            processed[new_col] = processed[old_col]
            
    # Convert square feet to square meters if needed
    if 'AREA_SQFT' in processed.columns:
        processed['area_sq_m'] = processed['AREA_SQFT'] * 0.092903
        
    return processed[['geometry', 'imperv_pct', 'area_sq_m']]


def process_land_use(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Process land use data and categorize
    
    Args:
        gdf: Raw land use GeoDataFrame
        
    Returns:
        Processed GeoDataFrame with simplified categories
    """
    processed = gdf.copy()
    
    # Map detailed land use codes to simplified categories
    landuse_mapping = {
        1: 'Urban',  # Residential
        2: 'Urban',  # Commercial
        3: 'Industrial',
        4: 'Transportation',
        5: 'Urban',  # Mixed use
        6: 'Forest',
        7: 'Agriculture',
        8: 'Recreation',
        9: 'Wetland',
        10: 'Water',
        11: 'Other'
    }
    
    # Apply mapping (adjust based on actual column names)
    if 'LUCODE' in processed.columns:
        processed['landuse'] = processed['LUCODE'].map(landuse_mapping)
    elif 'LU_CODE' in processed.columns:
        processed['landuse'] = processed['LU_CODE'].map(landuse_mapping)
        
    processed['area_sq_m'] = processed.geometry.area
    
    return processed[['geometry', 'landuse', 'area_sq_m']]


def process_ej_communities(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Process Environmental Justice communities data
    
    Args:
        gdf: Raw EJ communities GeoDataFrame
        
    Returns:
        Processed GeoDataFrame with vulnerability scores
    """
    processed = gdf.copy()
    
    # Calculate vulnerability score based on EJ criteria
    vulnerability_factors = []
    
    if 'INCOME' in processed.columns:
        processed['income_flag'] = processed['INCOME'].apply(lambda x: 1 if x == 'Y' else 0)
        vulnerability_factors.append('income_flag')
        
    if 'MINORITY' in processed.columns:
        processed['minority_flag'] = processed['MINORITY'].apply(lambda x: 1 if x == 'Y' else 0)
        vulnerability_factors.append('minority_flag')
        
    if 'ENGLISH' in processed.columns:
        processed['english_flag'] = processed['ENGLISH'].apply(lambda x: 1 if x == 'Y' else 0)
        vulnerability_factors.append('english_flag')
        
    # Calculate composite vulnerability score
    if vulnerability_factors:
        processed['vulnerability_score'] = processed[vulnerability_factors].sum(axis=1) / len(vulnerability_factors)
    else:
        processed['vulnerability_score'] = 0.5  # Default if no criteria available
        
    # Get population if available
    if 'POP' in processed.columns:
        processed['population'] = processed['POP']
    elif 'POPULATION' in processed.columns:
        processed['population'] = processed['POPULATION']
    else:
        processed['population'] = 0
        
    return processed[['geometry', 'vulnerability_score', 'population']]


def calculate_buffer_metrics(stations: gpd.GeoDataFrame, 
                            layer: gpd.GeoDataFrame,
                            buffer_distance: float,
                            metric_column: str,
                            aggregation: str = 'mean') -> pd.Series:
    """
    Calculate metrics within buffer distance of stations
    
    Args:
        stations: GeoDataFrame of monitoring stations
        layer: GeoDataFrame of the GIS layer to analyze
        buffer_distance: Buffer distance in meters
        metric_column: Column to calculate metrics from
        aggregation: Aggregation method ('mean', 'sum', 'max', 'min')
        
    Returns:
        Series of calculated metrics indexed by station
    """
    # Create buffers
    stations_buffered = stations.copy()
    stations_buffered['buffer'] = stations.geometry.buffer(buffer_distance)
    
    metrics = []
    
    for idx, station in stations_buffered.iterrows():
        # Find intersecting features
        buffer_geom = station['buffer']
        intersecting = layer[layer.intersects(buffer_geom)]
        
        if len(intersecting) > 0 and metric_column in intersecting.columns:
            if aggregation == 'mean':
                value = intersecting[metric_column].mean()
            elif aggregation == 'sum':
                value = intersecting[metric_column].sum()
            elif aggregation == 'max':
                value = intersecting[metric_column].max()
            elif aggregation == 'min':
                value = intersecting[metric_column].min()
            else:
                value = np.nan
        else:
            value = 0
            
        metrics.append(value)
        
    return pd.Series(metrics, index=stations.index)


def identify_upstream_areas(station_point: Point, 
                           watershed_gdf: gpd.GeoDataFrame,
                           elevation_raster_path: Optional[str] = None) -> Polygon:
    """
    Identify potential upstream contributing areas for a monitoring station
    
    Args:
        station_point: Point geometry of the station
        watershed_gdf: Watershed boundaries
        elevation_raster_path: Optional path to elevation raster for flow analysis
        
    Returns:
        Polygon representing upstream contributing area
    """
    # Find containing watershed
    containing_watershed = watershed_gdf[watershed_gdf.contains(station_point)]
    
    if len(containing_watershed) == 0:
        # If not in watershed, use nearest
        distances = watershed_gdf.geometry.distance(station_point)
        containing_watershed = watershed_gdf.loc[[distances.idxmin()]]
        
    watershed_geom = containing_watershed.geometry.iloc[0]
    
    if elevation_raster_path and os.path.exists(elevation_raster_path):
        # Use elevation to refine upstream area
        with rasterio.open(elevation_raster_path) as src:
            # Get elevation at station
            station_coords = [(station_point.x, station_point.y)]
            station_elev = list(src.sample(station_coords))[0][0]
            
            # Create mask of areas above station elevation
            # This is simplified - actual flow accumulation would be more complex
            watershed_mask = rasterio.features.geometry_mask(
                [watershed_geom],
                transform=src.transform,
                out_shape=(src.height, src.width),
                invert=True
            )
            
            elevation = src.read(1)
            upstream_mask = (elevation > station_elev) & watershed_mask
            
            # Convert mask to polygon
            # Simplified approach - would need flow accumulation for accuracy
            return watershed_geom  # Placeholder
    
    # Without elevation, use simplified approach
    # Create cone-shaped upstream area
    buffer_large = station_point.buffer(5000)  # 5km buffer
    buffer_small = station_point.buffer(500)   # 500m buffer
    
    # Create directional bias (simplified - assumes north is upstream)
    north_shift = Point(station_point.x, station_point.y + 2500)
    buffer_north = north_shift.buffer(3000)
    
    upstream_area = buffer_large.intersection(watershed_geom).union(buffer_north).intersection(watershed_geom)
    
    return upstream_area


def calculate_preservation_value(parcel_geom: Polygon,
                                station_location: Point,
                                current_landuse: str,
                                water_stress_score: float,
                                ej_flag: bool = False) -> Dict[str, float]:
    """
    Calculate the preservation value and potential impact of protecting a land parcel
    
    Args:
        parcel_geom: Geometry of the land parcel
        station_location: Location of affected monitoring station
        current_landuse: Current land use type
        water_stress_score: Current WSLUS score at station
        ej_flag: Whether station is in EJ community
        
    Returns:
        Dictionary of preservation metrics
    """
    # Calculate basic metrics
    area_hectares = parcel_geom.area / 10000
    distance = parcel_geom.centroid.distance(station_location)
    
    # Base preservation value by land type
    landuse_values = {
        'Forest': 100,
        'Wetland': 90,
        'Agriculture': 60,
        'Recreation': 50,
        'Urban': 30,
        'Industrial': 20,
        'Other': 40
    }
    
    base_value = landuse_values.get(current_landuse, 40)
    
    # Distance decay function (closer parcels have higher impact)
    distance_factor = np.exp(-distance / 2000)  # Decay with 2km scale
    
    # Size bonus (larger parcels provide more benefit)
    size_factor = np.log1p(area_hectares) / 5
    
    # Water stress amplifier
    stress_factor = water_stress_score / 50  # Normalize to ~0-2 range
    
    # EJ community bonus
    ej_multiplier = 1.5 if ej_flag else 1.0
    
    # Calculate composite score
    preservation_score = base_value * distance_factor * (1 + size_factor) * stress_factor * ej_multiplier
    
    # Estimate WSLUS reduction potential
    # This is a simplified model - would need calibration with real data
    landuse_impact = {
        'Forest': 0.15,
        'Wetland': 0.12,
        'Agriculture': 0.08,
        'Recreation': 0.06,
        'Urban': 0.04,
        'Industrial': 0.02,
        'Other': 0.05
    }
    
    impact_coefficient = landuse_impact.get(current_landuse, 0.05)
    estimated_wslus_reduction = impact_coefficient * area_hectares * distance_factor
    
    return {
        'preservation_score': preservation_score,
        'area_hectares': area_hectares,
        'distance_meters': distance,
        'estimated_wslus_reduction': estimated_wslus_reduction,
        'cost_efficiency': preservation_score / (area_hectares * 10000),  # Score per $10k (estimated)
        'ej_priority': ej_flag
    }


def generate_preservation_recommendations(stations_gdf: gpd.GeoDataFrame,
                                        opportunities_df: pd.DataFrame,
                                        budget_constraint: Optional[float] = None) -> Dict:
    """
    Generate specific preservation recommendations based on analysis
    
    Args:
        stations_gdf: Stations with WSLUS scores
        opportunities_df: Identified preservation opportunities
        budget_constraint: Optional budget limit in dollars
        
    Returns:
        Dictionary of recommendations
    """
    recommendations = {
        'immediate_actions': [],
        'short_term': [],
        'long_term': [],
        'policy_changes': [],
        'monitoring_gaps': []
    }
    
    # Identify critical areas
    critical_stations = stations_gdf[stations_gdf['WSLUS'] > 75]
    
    # Immediate actions for critical areas
    if len(critical_stations) > 0:
        recommendations['immediate_actions'].append({
            'action': 'Emergency stormwater intervention',
            'locations': critical_stations['station_id'].tolist()[:5],
            'rationale': 'Stations showing critical water stress levels',
            'estimated_cost': 500000
        })
    
    # Short-term preservation priorities
    if len(opportunities_df) > 0:
        top_opportunities = opportunities_df.nlargest(10, 'preservation_score')
        total_area = top_opportunities['area_hectares'].sum()
        
        recommendations['short_term'].append({
            'action': 'Land acquisition and preservation',
            'area_hectares': total_area,
            'parcels': len(top_opportunities),
            'estimated_impact': f"{top_opportunities['estimated_wslus_reduction'].sum():.1f} point WSLUS reduction",
            'estimated_cost': total_area * 50000  # $50k per hectare estimate
        })
    
    # Long-term strategies
    high_imperv_stations = stations_gdf[stations_gdf['impervious_stress'] > 0.7]
    if len(high_imperv_stations) > 0:
        recommendations['long_term'].append({
            'action': 'Green infrastructure retrofit program',
            'target_watersheds': len(high_imperv_stations),
            'rationale': 'Reduce impervious surface impact',
            'estimated_timeline': '5-10 years'
        })
    
    # Policy recommendations
    if stations_gdf['ej_vulnerability'].mean() > 0.5:
        recommendations['policy_changes'].append({
            'policy': 'Environmental Justice integration in watershed planning',
            'rationale': 'High correlation between EJ communities and water stress',
            'implementation': 'Require EJ impact assessment for all watershed projects'
        })
    
    # Monitoring gaps
    # Identify areas with high predicted stress but no stations
    recommendations['monitoring_gaps'].append({
        'recommendation': 'Deploy additional monitoring stations',
        'priority_areas': 'Industrial corridors and urban-rural interfaces',
        'number_needed': 10,
        'rationale': 'Improve spatial coverage of water quality monitoring'
    })
    
    return recommendations


def create_technical_report(analysis_results: Dict) -> str:
    """
    Generate a technical report summarizing the analysis
    
    Args:
        analysis_results: Dictionary containing all analysis outputs
        
    Returns:
        Markdown-formatted technical report
    """
    report = f"""
# Watershed Preservation Opportunity Map
## Technical Analysis Report
### Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## Executive Summary

This report presents the results of the Water Stress from Land Use Score (WSLUS) analysis,
identifying critical areas for watershed preservation in Massachusetts.

### Key Findings

- **Analyzed Stations**: {analysis_results.get('total_stations', 'N/A')}
- **Critical Risk Stations**: {analysis_results.get('critical_stations', 'N/A')}
- **Mean WSLUS Score**: {analysis_results.get('mean_wslus', 'N/A'):.1f}
- **Preservation Opportunities Identified**: {analysis_results.get('opportunities_count', 'N/A')}

## Methodology

### Water Stress from Land Use Score (WSLUS)

The WSLUS combines multiple stress indicators:

1. **Water Quality Stress** (60% weight)
   - Conductivity stress: Indicator of salt and runoff pollution
   - TDS stress: Total dissolved solids impact
   - pH stress: Acidity/alkalinity deviation
   - Temperature stress: Thermal pollution

2. **Land Use Stress** (40% weight)
   - Impervious surface coverage
   - Industrial/urban land use proximity
   - Forest/wetland coverage (negative stress)

### Spatial Analysis

Buffer zones analyzed:
- 500m: Immediate impact zone
- 1000m: Primary influence area
- 2000m: Extended watershed influence

## Results

### Stress Distribution

```
Risk Category    | Stations | Percentage
-----------------|----------|------------
Low (0-25)       | {analysis_results.get('low_risk', 0):8} | {analysis_results.get('low_risk_pct', 0):10.1f}%
Moderate (25-50) | {analysis_results.get('moderate_risk', 0):8} | {analysis_results.get('moderate_risk_pct', 0):10.1f}%
High (50-75)     | {analysis_results.get('high_risk', 0):8} | {analysis_results.get('high_risk_pct', 0):10.1f}%
Critical (75+)   | {analysis_results.get('critical_risk', 0):8} | {analysis_results.get('critical_risk_pct', 0):10.1f}%
```

### Environmental Justice Analysis

- Stations in EJ communities: {analysis_results.get('ej_stations', 'N/A')}
- Average WSLUS in EJ areas: {analysis_results.get('ej_mean_wslus', 'N/A'):.1f}
- Disparity ratio: {analysis_results.get('ej_disparity', 'N/A'):.2f}x

## Preservation Recommendations

### Immediate Actions (0-6 months)
{analysis_results.get('immediate_actions', 'To be determined based on analysis')}

### Short-term Priorities (6-24 months)
{analysis_results.get('short_term_priorities', 'To be determined based on analysis')}

### Long-term Strategy (2-10 years)
{analysis_results.get('long_term_strategy', 'To be determined based on analysis')}

## Model Performance

- Predictive Model R²: {analysis_results.get('model_r2', 'N/A'):.3f}
- Cross-validation Score: {analysis_results.get('cv_score', 'N/A'):.3f}
- RMSE: {analysis_results.get('rmse', 'N/A'):.2f}

## Data Quality Notes

- Temporal coverage: {analysis_results.get('date_range', 'N/A')}
- Spatial coverage: {analysis_results.get('spatial_coverage', 'N/A')}
- Missing data: {analysis_results.get('missing_data_pct', 'N/A'):.1f}%

## Appendix: Technical Specifications

- Coordinate System: NAD83 / Massachusetts Mainland (EPSG:26986)
- Analysis Software: Python 3.9+ with GeoPandas, Scikit-learn
- Data Sources: MassGIS, MassDEP Water Quality Database

---

*This report is generated automatically by the WSLUS Analysis System*
"""
    
    return report


# Example usage
if __name__ == "__main__":
    # Initialize data loader
    loader = MassGISDataLoader()
    
    # Load sample data
    print("Loading MassGIS data layers...")
    
    try:
        # Load impervious surface
        impervious = loader.load_layer('impervious')
        impervious_processed = process_impervious_surface(impervious)
        print(f"✓ Loaded impervious surface: {len(impervious_processed)} features")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Note: Update MASSGIS_URLS with actual data URLs")
    
    print("\nMassGIS utilities module loaded successfully!")
    print("Use MassGISDataLoader class to download and process GIS data")
