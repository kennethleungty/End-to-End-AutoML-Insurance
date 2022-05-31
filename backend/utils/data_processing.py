import h2o
import json

def separate_id_col(h2o_frame):
    """ Separate ID column (if any) and return dataset with non-ID columns

    Args:
        h2o_frame ([H2O frame]): Dataset in H2O frame format

    Returns:
        id_name: Name of the ID column (depending on which one is present)
        X_id: H2O frame of the ID array
        X_h2o: Return X dataset (without ID column). Same as input H2O frame if no ID present
    """
    # List of possible ID column names
    possible_id_list = ['ID', 'Id', 'id']

    for i in possible_id_list:
        if i in h2o_frame.names:
            id_name = i
            X_id = h2o_frame[:, id_name]
            X_h2o = h2o_frame.drop(id_name)
            break
        else:
            id_name, X_id = None, None
            X_h2o = h2o_frame

    return id_name, X_id, X_h2o


def match_col_types(h2o_frame):
    """[summary]

    Args:
        h2o_frame: Input test set H2O frame

    Returns:
        h2o_frame: Returns H2O frame with column types matching that of train set
    """
    # Load dictionary of train set column types
    with open('data/processed/train_col_types.json') as f:
        train_col_types = json.load(f)

    # Match test set column types with train set
    for key in train_col_types.keys():
        try:
            # If col types do not match, convert test set col type to match train set   
            if train_col_types[key] != h2o_frame.types[key]:
                if train_col_types[key] == 'real' and h2o_frame.types[key] == 'enum':
                    h2o_frame[key] = h2o_frame[key].ascharacter().asnumeric()
                elif train_col_types[key] == 'real':
                    h2o_frame[key] = h2o_frame[key].asnumeric()
                elif train_col_types[key] == 'int':
                    h2o_frame[key] = h2o_frame[key].asfactor()
                elif train_col_types[key] == 'str':
                    h2o_frame[key] = h2o_frame[key].ascharacter()
        except:
            pass

    return h2o_frame

