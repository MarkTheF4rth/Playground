import copy
import json

from .data_source import DataSource

SOURCE_1_DATA = {
    "1_id1": {
        "attr1": "1.1",
        "attr2": "1.2",
        "attr4": "1.3"
    },
    "1_id2": {
        "attr1": "1.1",
        "attr4": "1.3"
    }
}


class Source1(DataSource):
    def __init__(self):
        self.source_name = "source1"
        self.attribute_wishlist = {
            "attr1": "source1",
            "attr2": "source2",
            "attr3": "source2",
            "attr4": "source1"
        }
        self.inmemory_data = copy.copy(SOURCE_1_DATA)

    def get_data(self, common_id):
        return SOURCE_1_DATA[self._get_local_id(common_id)]

    def set_data(self, common_id, attribute_dict):
        self.inmemory_data[self._get_local_id(common_id)] = attribute_dict

    def _get_local_id(self, common_id):
        return f"1_{common_id}"

    def debug_output(self):
        print("SOURCE 1:", json.dumps(self.inmemory_data, indent=4))
