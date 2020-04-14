import plotly.express as px
import pandas as pd
import json

def plot_county_data(df, yname, yrange, xname='fips'):
    with open('geojson/geojson-counties-fips.json') as f:
        counties = json.load(f)

    fig = px.choropleth_mapbox(df, geojson=counties, locations=xname, color=yname,
                               color_continuous_scale="Viridis",
                               range_color=yrange,
                               mapbox_style="carto-positron",
                               zoom=3, center = {"lat": 37.0902, "lon": -95.7129},
                               opacity=0.5,
                               labels={'unemp':'unemployment rate'}
                              )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig