import folium


def create_choropleth_map(canton_amount_df):
    """
    :param canton_amount_df: the data frame of all cantons with their sum of approved amounts
    :return: the choropleth overlayed map
    """
    cantons_topo_path = r'json/ch-cantons.topojson.json'

    # We create a map centered to Switzerland
    swiss_map = folium.Map(location=[47.05016819999999, 8.309307200000035],
                           tiles='Mapbox Bright', zoom_start=7)

    # We colorize it using the canton_amount dataframe
    # Note: we asigned tresholds manually, to create a better visual.
    swiss_map.choropleth(geo_path=cantons_topo_path,
                         data=canton_amount_df,
                         columns=['Canton', 'Approved Amount'],
                         key_on='feature.id',
                         threshold_scale=[1e+04, 4.0e+08, 4.6e+08, 1.5e+09, 2.3e+09,
                                          4e+09],
                         topojson='objects.cantons',
                         fill_color='YlGn',
                         fill_opacity=100,
                         line_opacity=1000,
                         legend_name='Total Amount of Grants'
                         )
    return swiss_map


