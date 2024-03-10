import pandas as pd
import numpy as np

def add_columns(data, col, sequence=[1, 3, 5, 10]):
    shifted = pd.DataFrame(columns=[f'{col}+{x}' for x in sequence], index=data.index)
    
    for i, (idx, row) in enumerate(data[[col]].iterrows()):
        vals = []
        
        for ix in sequence:
            try:
                ixs = i + ix
                # print(ixs)
                
                if ixs > len(data):
                    vals.append(np.nan)
                    # break
                else:
                    val = data.iloc[ixs][col]
                    vals.append(val)
            except IndexError:
                vals.append(np.nan)
            except ZeroDivisionError:
                vals.append(np.nan)
        
        shifted.iloc[i, :] = vals
    
    return shifted


def add_all_cols(data):
    new_df = pd.DataFrame(index=data.index)

    for col in data.columns:
        new_df = pd.concat([new_df, data[col]], axis=1)
        new_df = pd.concat([new_df, add_columns(data, col)], axis=1)

    return new_df