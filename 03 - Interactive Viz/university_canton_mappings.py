import requests
import json

def get_canton_id(google_url, university):
    """
    Getting the canton corresponding to university
    :param university:
    :return:
    """
    json_data = requests.get(google_url, params={"address": university}).json()
    
    # We try to return the first result with 'administrative_area_level_1'
    for component in json_data['results'][0]['address_components']:
        if component['types'][0] == 'administrative_area_level_1':
            return component['short_name']
        
    # Else, we return empty and deal with that later in the main code.
    return ''


def get_cantons_json(json_path):
    """
    Creates a dictionary of (canton_name->canton_id) out of JSON file on the 'json_path'.
    :param json_path:
    :return:
    """
    with open(json_path) as json_data:
        cantons_data = json.load(json_data)

    cantons_mappings_dict = {}
    for canton in cantons_data["objects"]["cantons"]["geometries"]:
        cantons_mappings_dict[canton["properties"]["name"]] = canton["id"]

    return cantons_mappings_dict



