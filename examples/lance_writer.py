import lance


def write_batch_to_lance(arrow_table, output_path):
    lance.write_dataset(arrow_table, output_path, mode="overwrite")
