
def get_unique_cell_names(_type, cell_types, labels):
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
    else:
        raise TypeError()
    return cell_names
