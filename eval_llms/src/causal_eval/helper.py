import re
import numpy as np

def get_adjacency_matrice(file_dir):
    pattern_node = r'V\d+'
    pattern_bt = r'bt\d'
    bt_id = 0
    adj = np.zeros((10,10))

    adj_ls = []
    # Open the file in read mode
    with open(file_dir, 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Process the line (strip is used to remove the newline character)
            text = line.strip()
            line = re.sub(r'[^a-zA-Z0-9]', '', text)

            # Find all matches in the text
            bt_ind = re.findall(pattern_bt, line)

            # Print the extracted matches
            
            if len(bt_ind) > 0:
                adj_ls.append(adj)
                adj = np.zeros((10,10))
                bt_id += 1
            else:
                neighbors = re.findall(pattern_node, line)
                for i in range(len(neighbors)):
                    if i == 0:
                        ni = re.sub(r'[^0-9]', '', neighbors[i])
                        row_index = int(ni)
                    else: 
                        ni = re.sub(r'[^0-9]', '', neighbors[i])
                        col_index = int(ni)
                        adj[row_index,col_index]  = 1
    adj_ls.append(adj)
    return adj_ls[1:] 

