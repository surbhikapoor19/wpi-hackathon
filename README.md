# Watershed Preservation Opportunity Map

A data-driven decision support system for identifying priority areas for watershed conservation in Massachusetts, combining water quality monitoring data with **real economic data** from MA DOR and spatial information.

## Quick Start

1. **Run the Analysis**:
   ```bash
   jupyter notebook watershed_preservation_model.ipynb
   ```
   Execute all cells to generate WSLUS scores and preservation opportunities.

2. **Launch Dashboard**:
   ```bash
   streamlit run watershed_streamlit_app.py
   ```
   Navigate to the "WSLUS Model Insights" tab for interactive visualizations.

## What This Project Does

**Problem**: Massachusetts needs to prioritize limited conservation resources to maximize water quality improvements, especially in Environmental Justice communities.

**Solution**: Creates a **Water Stress from Land Use Score (WSLUS)** that:
- Quantifies water quality stress from multiple parameters
- Integrates **real economic data** from MA DOR Municipal Databank
- Identifies high-impact preservation opportunities
- Provides actionable recommendations for government stakeholders

## Key Features

- **Multi-Parameter Water Quality Analysis**: pH, conductivity, TDS, temperature, dissolved oxygen
- **Real Economic Data Integration**: Uses MA DOR municipal income data (not synthetic!)
- **Environmental Justice Focus**: Prioritizes EJ communities in scoring
- **Machine Learning Model**: Predicts water stress from land characteristics
- **Interactive Visualizations**: Maps, charts, and priority matrices
- **Actionable Recommendations**: Ranked preservation opportunities with impact scores

## Data Sources

### Water Quality Data
- **Source**: Sample dataset (`sample_water_quality.csv`)
- **Note**: Demonstration data - replace with real monitoring data in production

### Economic Data
- **Source**: **Massachusetts DOR Municipal Databank** (real data!)
- **URL**: https://dls-gw.dor.state.ma.us/reports/rdPage.aspx?rdReport=Socioeconomic.MedHouseholdFamInc
- **Data Type**: Municipality-level median household income
- **Matching**: Extracts municipality from location descriptions or geocodes from coordinates
- **Fallback**: Synthetic proxies if DOR data unavailable
- **Indicators**: Income, poverty, population density, education, unemployment, home values

### GIS Data
- **Source**: MassGIS layers (when available) or economic proxies
- **Note**: Original GIS layers found to be sparse - system uses economic proxies as alternative

## Output Files

- `stations_with_wslus_scores.csv` - Complete station analysis
- `preservation_opportunities.csv` - Ranked conservation targets
- `wslus_predictive_model.pkl` - Trained ML model
- `executive_summary.json` - Key findings
- `watershed_preservation_map.html` - Interactive map

## Documentation

For comprehensive documentation, see **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)** which includes:
- Detailed problem statement
- Complete data source descriptions
- Methodology explanation
- Technical architecture
- Use cases and limitations
- Future enhancement roadmap

For DOR data integration details, see **[DOR_DATA_INTEGRATION.md](DOR_DATA_INTEGRATION.md)**

## Important Notes

✅ **Real Economic Data**: The system now uses **real MA DOR municipal income data** (not synthetic!)

⚠️ **Sample Data**: Water quality data is demonstration data. Replace with actual monitoring data for real-world applications.

⚠️ **Partial Coverage**: Built-in DOR dataset includes common municipalities. For full coverage, download complete dataset from DOR website.

## Requirements

- Python 3.8+
- pandas, numpy, geopandas, scikit-learn, plotly, folium, streamlit
- Jupyter Notebook

## License

Designed for watershed conservation and environmental justice applications.
