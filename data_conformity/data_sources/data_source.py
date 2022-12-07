class DataSource:
    source_name: str
    attribute_wishlist: dict

    def get_all_ids(self):
        raise NotImplemented

    def get_data(self, common_id):
        raise NotImplemented

    def set_data(self, common_id, attribute_dict):
        raise NotImplemented
