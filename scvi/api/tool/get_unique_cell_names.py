import numpy as np


def get_unique_cell_names(_type, cell_types, labels):
    """
    get unique cell names
     :param _type: _type has two choice: 1. "type_order": labels are "cell_types[label] {identifier}"
                                        2. "cell_order": labels are "cell {identifier}"
                                        3. "manual": labels are cell_names
    :param cell_types: cell types
    :param labels: :type: Union[list[str], list[int]] labels
    :return: list[str]
    """
    if isinstance(labels[0], str):
        labels = np.array(list(map(lambda x: cell_types.index(x), labels)))

    if _type == "cell_order":
        cell_names = ['cell ' + str(i + 1) for i in range(len(labels))]
    elif _type == "type_order":
        label_dict = {cell_type: 0 for cell_type in cell_types}
        cell_names = []
        labels = labels.flatten()
        for i in range(len(labels)):
            cell_type = cell_types[labels[i]]
            cell_names.append(cell_type + " " + str(label_dict[cell_type] + 1))
            label_dict[cell_type] = label_dict[cell_type] + 1
    elif _type == "manual":
        cell_names = labels
    else:
        raise TypeError()
    return cell_names
