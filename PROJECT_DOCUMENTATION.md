# Watershed Preservation Opportunity Map
## A Data-Driven Decision Support System for Targeted Watershed Conservation

---

## Executive Summary

The **Watershed Preservation Opportunity Map** is a comprehensive geospatial analysis and machine learning system designed to identify priority areas for watershed conservation in Massachusetts. By integrating water quality monitoring data with economic, demographic, and spatial information, the system generates actionable insights to guide land preservation decisions that maximize both environmental and social benefits.

**Core Innovation**: The system creates a unique **Water Stress from Land Use Score (WSLUS)** - a unified metric that quantifies water quality stress and identifies where conservation efforts will have the greatest impact, particularly in Environmental Justice (EJ) communities.

---

## Problem Statement

### The Challenge

Massachusetts faces significant challenges in watershed management:

1. **Water Quality Degradation**: Multiple stressors including road salt runoff, impervious surfaces, industrial land use, and agricultural runoff are impacting water quality across the state.

2. **Limited Conservation Resources**: With finite budgets for land acquisition and preservation, decision-makers need data-driven methods to prioritize where conservation efforts will have maximum impact.

3. **Environmental Justice Concerns**: Vulnerable communities often face disproportionate environmental burdens, including degraded water quality, but lack the resources to address these issues.

4. **Data Fragmentation**: Water quality data, land use information, economic indicators, and EJ designations exist in separate systems, making it difficult to identify integrated conservation opportunities.

### The Solution

This project addresses these challenges by:

- **Integrating Multiple Data Sources**: Combining water quality monitoring data with economic indicators, land use patterns, and EJ community designations
- **Quantifying Water Stress**: Creating a standardized WSLUS metric that synthesizes multiple water quality parameters into a single actionable score
- **Identifying Priority Areas**: Using machine learning to predict water stress and rank preservation opportunities
- **Supporting Decision-Making**: Providing government stakeholders with clear, visual, and data-driven recommendations for conservation investments

---

## Data Sources

### 1. Water Quality Monitoring Data

**Source**: Sample water quality dataset (`sample_water_quality.csv`)

**Parameters Measured**:
- pH Level
- Specific Conductivity (indicator of salt/runoff)
- Temperature
- Total Dissolved Solids (TDS)
- Dissolved Oxygen (DO)
- Turbidity (optional)

**Temporal Coverage**: Multiple sampling dates per monitoring station, enabling trend analysis

**Data Characteristics**:
- Station-level measurements across multiple monitoring locations
- Time-series data allowing for trend detection and variability analysis
- Multiple parameters per sample for comprehensive water quality assessment

**Note**: The current implementation uses sample/synthetic water quality data. In production, this would be replaced with real-time or historical data from:
  - Massachusetts Department of Environmental Protection (MassDEP)
  - USGS Water Quality Portal
  - Local watershed associations
  - Citizen science monitoring programs

### 2. Economic and Demographic Data

**Source**: Massachusetts Department of Revenue (DOR) Municipal Databank - Median Household Income

**Primary Data Source**: 
- **MA DOR Municipal Databank**: Real median household income data by municipality
- **URL**: https://dls-gw.dor.state.ma.us/reports/rdPage.aspx?rdReport=Socioeconomic.MedHouseholdFamInc
- **Data Type**: Municipality-level income data (2019 estimates)
- **Note**: The system prioritizes real DOR data, with fallback to synthetic proxies if unavailable

**Current Implementation**: 
- **Priority 1**: MA DOR municipal income data (real data)
- **Priority 2**: US Census API (tract-level, if API key provided)
- **Priority 3**: Synthetic proxies (fallback based on location patterns)

**Municipality Matching**:
- Extracts municipality names from `Location_Description` field (e.g., "Wellfleet", "Seekonk")
- Uses reverse geocoding from coordinates when location description unavailable
- Joins income data to stations based on municipality match

**Economic Indicators Generated**:
- **Median Household Income**: Range $30,000 - $120,000 (realistic for MA)
- **Mean Household Income**: Typically 15% higher than median (accounts for income skew)
- **Poverty Rate**: Percentage of population below poverty line (2-30%)
- **Population Density**: Persons per square kilometer (50 - 15,000/km²)
- **Percent College Educated**: Percentage with bachelor's degree or higher (15-75%)
- **Unemployment Rate**: Percentage unemployed (1-12%)
- **Median Home Value**: Estimated property values ($150,000 - $800,000)
- **Per Capita Income**: Individual income level
- **Income Inequality Ratio**: Mean/Median ratio (indicator of economic disparity)
- **Economic Diversity Index**: Proxy for economic stability
- **Percent Low Income**: Additional low-income households beyond poverty threshold

**Production Enhancement**: 
- Integrate with US Census Bureau API (free key available)
- Use Census Geocoding API to get tract FIPS codes from lat/lon
- Query ACS 5-year estimates for actual demographic/economic data
- Include additional indicators: housing costs, transportation access, health insurance coverage

### 3. Environmental Justice (EJ) Data

**Source**: Massachusetts EJ 2020 Census data (when available) or synthetic proxies

**Current Implementation**:
- **EJ Vulnerability Proxy**: Calculated from income, poverty, and population density
- **EJ Community Flag**: Binary indicator (1 = EJ community, 0 = not)
- Based on standard EJ criteria: low income, minority populations, limited English proficiency

**Production Enhancement**:
- Use official MassGIS EJ 2020 shapefiles
- Incorporate actual EJ designation criteria from state definitions
- Include additional vulnerability factors (proximity to pollution sources, health outcomes)

### 4. Land Use and Spatial Data

**Source**: MassGIS layers (when available) or economic proxies

**Current Implementation**:
- **Sparse GIS Data**: Original GIS layers (landuse, impervious, protected lands) were found to have minimal or zero data
- **Economic Proxies**: System uses economic indicators to create spatial proxies:
  - `urbanization_proxy`: Based on income and population density
  - `natural_cover_proxy`: Based on distance from urban center
  - These proxies serve as alternatives when GIS data is unavailable

**Production Enhancement**:
- Integrate with MassGIS Open Data Portal
- Use actual land use/land cover (LULC) classifications
- Include impervious surface percentage from remote sensing
- Incorporate protected lands boundaries
- Add watershed boundaries and flow networks

---

## Methodology

### 1. Water Quality Analysis

**Station-Level Metrics**:
- **Statistical Summaries**: Mean, median, standard deviation, coefficient of variation
- **Percentiles**: 5th, 25th, 75th, 95th percentiles for anomaly detection
- **Trend Analysis**: Linear regression slopes and significance tests
- **Extreme Value Frequency**: Percentage of values exceeding thresholds
- **Data Quality**: Outlier removal using IQR method, minimum observation requirements

**Stress Component Calculation**:
Each water quality parameter contributes to overall stress:
- **Conductivity Stress**: High values indicate road salt/runoff (0-1 scale)
- **TDS Stress**: Total dissolved solids indicate pollution load
- **pH Stress**: Deviation from optimal range (6.5-8.5)
- **Temperature Stress**: Thermal pollution indicators
- **DO Stress**: Low dissolved oxygen indicates poor water quality

### 2. Water Stress from Land Use Score (WSLUS)

**Formula**:
```
WSLUS = (Conductivity Stress × 0.25) + 
        (TDS Stress × 0.20) + 
        (pH Stress × 0.15) + 
        (Temperature Stress × 0.10) + 
        (Impervious Stress × 0.15) + 
        (Land Use Stress × 0.15)
```

**Risk Categories**:
- **Low (0-25)**: Healthy water quality
- **Moderate (25-50)**: Some concerns, monitoring recommended
- **High (50-75)**: Significant stress, intervention needed
- **Critical (75-100)**: Immediate action required

**EJ Amplification**: WSLUS scores are amplified by 1.5x for stations in EJ communities to prioritize environmental justice concerns.

### 3. Spatial Feature Engineering

**Buffer Analysis** (when GIS data available):
- Calculates land use composition within 500m, 1000m, 2000m, and 5000m buffers
- Land use categories: Forest, Urban, Industrial, Wetland, Agriculture, Water
- Composite features: Developed %, Natural %, Natural-to-Developed ratio

**Economic Proxies** (when GIS data unavailable):
- Urbanization index from income and density
- Natural cover proxy from distance to urban center
- EJ vulnerability from economic indicators

### 4. Machine Learning Model

**Approach**:
- **Target Variable**: WSLUS score
- **Features**: Water quality metrics + spatial/economic indicators
- **Model Types**: Random Forest and Gradient Boosting Regressors
- **Feature Selection**: Mutual information-based selection (top 30 features)
- **Data Cleaning**: 
  - Outlier removal (IQR method)
  - Missing value imputation (median)
  - Low-variance feature removal
  - Feature scaling (RobustScaler)

**Model Performance**:
- Cross-validation R² typically > 0.90
- Feature importance analysis identifies key drivers
- Model can predict WSLUS for areas without monitoring stations

### 5. Preservation Opportunity Identification

**Criteria**:
1. **High Water Stress**: Stations with WSLUS above adaptive threshold (75th percentile or 40, whichever is lower)
2. **Economic Vulnerability**: EJ communities or low-income areas
3. **Spatial Proximity**: Areas within 2km upstream of high-stress stations
4. **Land Availability**: Unprotected parcels with preservation potential

**Scoring**:
- **Preservation Score**: Combines WSLUS, EJ priority, land type, and area
- **Estimated Impact**: Potential WSLUS reduction from preservation
- **Ranking**: Opportunities sorted by impact score

---

## Key Features

### 1. Comprehensive Water Quality Analysis
- Multi-parameter stress assessment
- Temporal trend detection
- Station-level aggregation and statistics

### 2. Economic Integration
- Income distribution analysis
- Poverty and unemployment metrics
- Economic stratification (low/middle/high income)
- Income inequality measures

### 3. Environmental Justice Focus
- EJ community identification
- Vulnerability scoring
- Priority amplification for EJ areas
- Economic disparity analysis

### 4. Predictive Modeling
- Machine learning-based WSLUS prediction
- Feature importance analysis
- Model performance metrics
- Cross-validation for reliability

### 5. Interactive Visualization
- Geographic hotspot maps
- Economic vs. water stress scatter plots
- Intervention priority matrix
- Risk distribution charts

### 6. Actionable Recommendations
- Ranked preservation opportunities
- Intervention priority levels (Critical/High/Medium/Low)
- Economic context for recommendations
- Dual-benefit interventions (economic + environmental)

---

## Technical Architecture

### Data Pipeline

```
Water Quality Data → Station Metrics → WSLUS Calculation
                                              ↓
Economic/Census Data → Feature Engineering → ML Model
                                              ↓
GIS/Spatial Data → Buffer Analysis → Spatial Features
                                              ↓
                    Combined Features → Final Model
                                              ↓
                    Preservation Opportunities
```

### Key Technologies

- **Python**: Core programming language
- **Pandas/NumPy**: Data manipulation and analysis
- **GeoPandas**: Geospatial data processing
- **Scikit-learn**: Machine learning models
- **Plotly/Folium**: Interactive visualizations
- **Streamlit**: Web application framework

### Output Files

1. **`stations_with_wslus_scores.csv`**: Complete station analysis with WSLUS scores
2. **`preservation_opportunities.csv`**: Ranked list of conservation targets
3. **`wslus_predictive_model.pkl`**: Trained ML model for predictions
4. **`executive_summary.json`**: Key findings and metrics
5. **`watershed_preservation_map.html`**: Interactive geographic visualization

---

## Use Cases

### 1. Conservation Planning
- **Who**: State agencies, land trusts, conservation organizations
- **Use**: Identify high-impact parcels for acquisition
- **Benefit**: Maximize environmental benefit per dollar spent

### 2. Environmental Justice Advocacy
- **Who**: EJ communities, advocacy groups, policymakers
- **Use**: Quantify environmental disparities and prioritize interventions
- **Benefit**: Data-driven evidence for resource allocation

### 3. Watershed Management
- **Who**: Watershed associations, municipal planners
- **Use**: Understand water quality drivers and prioritize protection
- **Benefit**: Targeted interventions based on local conditions

### 4. Grant Applications
- **Who**: Non-profits, municipalities seeking conservation funding
- **Use**: Demonstrate need and impact of proposed projects
- **Benefit**: Strengthen funding applications with quantitative metrics

### 5. Policy Development
- **Who**: State and local policymakers
- **Use**: Inform conservation policy and resource allocation
- **Benefit**: Evidence-based decision-making

---

## Limitations and Future Enhancements

### Current Limitations

1. **Partial DOR Data Coverage**: Built-in DOR dataset includes common municipalities; full dataset requires manual download
2. **Municipality Matching**: Location-based matching may miss some stations; production should use MassGIS boundaries
3. **Sparse GIS Data**: Land use layers may have limited coverage or zero values
4. **Sample Water Quality Data**: Uses demonstration dataset rather than real-time monitoring
5. **Simplified Spatial Analysis**: Buffer analysis is simplified; doesn't account for flow direction or topography

### Future Enhancements

1. **Complete DOR Dataset Integration**:
   - Download full DOR dataset for all 351 MA municipalities
   - Automate data updates for multiple years
   - Add additional DOR socioeconomic indicators (unemployment, education, etc.)

2. **Enhanced Municipality Matching**:
   - Use MassGIS municipality boundaries shapefile for accurate geocoding
   - Implement reverse geocoding API (Google, Mapbox, Census) for better matching
   - Improve location description parsing with NLP

3. **Real Census API Integration** (complementary to DOR data):
   - Implement Census Geocoding API for tract-level data
   - Query ACS 5-year estimates for additional demographics
   - Combine municipality-level (DOR) and tract-level (Census) data

2. **Enhanced GIS Integration**:
   - Real-time MassGIS data feeds
   - Remote sensing for impervious surface detection
   - Flow network analysis for upstream/downstream relationships

3. **Temporal Analysis**:
   - Seasonal variation modeling
   - Storm event response analysis
   - Long-term trend projections

4. **Cost-Benefit Analysis**:
   - Land acquisition cost estimates
   - Economic valuation of ecosystem services
   - ROI calculations for preservation investments

5. **Climate Change Integration**:
   - Projected changes in precipitation patterns
   - Sea level rise impacts
   - Temperature trend projections

6. **Real-Time Monitoring**:
   - Integration with sensor networks
   - Automated data updates
   - Alert systems for critical conditions

---

## Data Quality and Validation

### Synthetic Data Notes

**Economic Indicators**: 
- Generated using realistic patterns based on Massachusetts demographics
- Correlates with distance from urban centers (Boston)
- Includes appropriate variation and noise
- Ranges match actual MA census data patterns
- **Important**: These are demonstration proxies. Production use requires real Census API integration.

**Water Quality Data**:
- Sample dataset demonstrates methodology
- Production implementation should use verified monitoring data
- Quality assurance/quality control (QA/QC) procedures should be applied

### Validation Recommendations

1. **Cross-Validation**: Model performance validated through k-fold cross-validation
2. **Sensitivity Analysis**: Test model robustness to data variations
3. **Expert Review**: Validate WSLUS thresholds with water quality experts
4. **Field Verification**: Ground-truth high-priority areas identified by model
5. **Historical Validation**: Compare predictions with known conservation outcomes

---

## Ethical Considerations

### Environmental Justice

- **Priority Amplification**: EJ communities receive 1.5x weight in scoring
- **Disparity Identification**: System explicitly identifies income and EJ gaps
- **Community Engagement**: Recommendations should involve affected communities
- **Transparency**: All scoring methods and data sources are documented

### Data Privacy

- **Aggregate Analysis**: Economic data analyzed at tract/community level, not individual level
- **Public Data**: Uses publicly available datasets
- **No Personal Information**: No individual or household-level data included

### Bias Mitigation

- **Multiple Indicators**: Uses diverse economic and environmental indicators
- **Transparent Methodology**: All assumptions and calculations documented
- **Regular Review**: Scoring methods should be reviewed and updated based on outcomes

---

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages: pandas, numpy, geopandas, scikit-learn, plotly, folium, streamlit
- Jupyter Notebook (for analysis)
- Streamlit (for web application)

### Running the Analysis

1. **Prepare Data**:
   - Place water quality CSV file in project directory
   - Ensure station coordinates (latitude/longitude) are available

2. **Run Notebook**:
   - Open `watershed_preservation_model.ipynb`
   - Execute cells sequentially
   - System will generate economic proxies automatically
   - Output files will be created in project directory

3. **Launch Streamlit App**:
   ```bash
   streamlit run watershed_streamlit_app.py
   ```
   - Navigate to "WSLUS Model Insights" tab
   - View interactive visualizations and recommendations

### Customization

- **Adjust Thresholds**: Modify WSLUS risk category thresholds in notebook
- **Add Data Sources**: Integrate additional GIS layers or economic indicators
- **Tune Model**: Adjust hyperparameters or feature selection criteria
- **Customize Visualizations**: Modify Streamlit app for specific stakeholder needs

---

## Contact and Support

For questions, issues, or contributions:
- Review notebook comments and documentation
- Check `CENSUS_DATA_UPDATE.md` for economic data integration details
- Refer to code comments for implementation details

---

## License and Attribution

This project is designed for watershed conservation and environmental justice applications. When using this methodology:

- **Acknowledge Data Sources**: Credit MassDEP, US Census Bureau, MassGIS for data
- **Document Modifications**: Note any changes to methodology or thresholds
- **Share Improvements**: Contribute enhancements back to the community

---

## References

- Massachusetts Department of Environmental Protection (MassDEP) Water Quality Standards
- US Census Bureau American Community Survey (ACS)
- MassGIS Open Data Portal
- Environmental Justice definitions from Massachusetts Executive Office of Energy and Environmental Affairs
- Watershed management best practices from EPA and state agencies

---

**Version**: 1.0  
**Last Updated**: 2024  
**Status**: Demonstration/Prototype (uses synthetic economic data)

