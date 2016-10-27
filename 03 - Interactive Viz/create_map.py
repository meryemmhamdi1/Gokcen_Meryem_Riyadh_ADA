import folium


def create_choropleth_map(canton_amount_df):
    """
    :param canton_amount_df: the data frame of all cantons with their sum of approved amounts
    :return: the choropleth overlayed map
    """
    cantons_topo_path = r'json/ch-cantons.topojson.json'
    swiss_map = folium.Map(location=[47.05016819999999, 8.309307200000035],
                       tiles='Mapbox Bright', zoom_start=7)
    swiss_map.choropleth(geo_path=cantons_topo_path, topojson='objects.cantons')

    swiss_map = folium.Map(location=[47.05016819999999, 8.309307200000035],
                           tiles='Mapbox Bright', zoom_start=7)

    swiss_map.choropleth(geo_path=cantons_topo_path,
                         data=canton_amount_df,
                         columns=['Canton', 'Approved Amount'],
                         key_on='feature.id',
                         threshold_scale=[0.000000e+00, 1.766910e+05, 9.119410e+07, 4.590737e+08, 1.877102e+09,
                                          3.627465e+09],
                         topojson='objects.cantons',
                         fill_color='YlGn',
                         fill_opacity=100,
                         line_opacity=1000,
                         legend_name='Total Amount of Grants'
                         )
    return swiss_map


