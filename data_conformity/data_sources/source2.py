import json
import copy

from .data_source import DataSource

SOURCE_2_DATA = {
    "2_id1": {
        "attr1": "2.2",
        "attr2": "2.2",
        "attr3": "2.3"
    },
    "2_id2": {
        "attr2": "2.2",
        "attr3": "2.3"
    }
}


class Source2(DataSource):
    def __init__(self):
        self.source_name = "source2"
        self.attribute_wishlist = {
            "attr1": "source1",
            "attr2": "source2",
            "attr4": "source1"
        }
        self.inmemory_data = copy.copy(SOURCE_2_DATA)

    def get_all_ids(self):
        return [x.lstrip('2_') for x in SOURCE_2_DATA.keys()]

    def get_data(self, common_id):
        return SOURCE_2_DATA[self._get_local_id(common_id)]

    def set_data(self, common_id, attribute_dict):
        self.inmemory_data[self._get_local_id(common_id)] = attribute_dict

    def _get_local_id(self, common_id):
        return f"2_{common_id}"

    def debug_output(self):
        print("SOURCE 2:", json.dumps(self.inmemory_data, indent=4))
