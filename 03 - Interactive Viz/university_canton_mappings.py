import requests
import json

def get_canton_id(google_url, university):
    """
    Getting the canton corresponding to university
    :param university:
    :return:
    """
    json_data = requests.get(google_url, params={"address": university}).json()
    i = 0
    canton = ''
    for a in json_data['results'][0]['address_components']:
        if (json_data['results'][0]['address_components'][i]['types'][0] == 'administrative_area_level_1'):
            canton = json_data['results'][0]['address_components'][i]['short_name']
            break
        i += 1

    return canton


def get_cantons_json(json_path):
    """

    :param json_path:
    :return:
    """
    with open(json_path) as json_data:
        cantons_data = json.load(json_data)

    cantons_mappings = []
    for i in range(0, len(cantons_data["objects"]["cantons"]["geometries"])):
        canton_id = cantons_data["objects"]["cantons"]["geometries"][i]["id"]
        canton_name = cantons_data["objects"]["cantons"]["geometries"][i]["properties"]["name"]
        cantons_mappings.append((canton_name, canton_id))

    cantons_mappings_dict = dict(cantons_mappings)
    cantons_mappings_dict.values()
    return cantons_mappings_dict



