import data_sources

if __name__ == "__main__":
    local_source_list = {}
    for source in data_sources.source_list:
        source_instance = source()
        local_source_list[source_instance.source_name] = source_instance
    id_list_source = local_source_list['source2']

    for common_id in id_list_source.get_all_ids():
        for source in local_source_list.values():
            attr_build = {}
            for name, attr_source in source.attribute_wishlist.items():
                attr_build[name] = local_source_list[attr_source].get_data(common_id)[name]

            source.set_data(common_id, attr_build)

    for source in local_source_list.values():
        source.debug_output()
