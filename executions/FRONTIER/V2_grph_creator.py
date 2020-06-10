#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

# base_data = pd.read_csv('../NLPWorkspace/executions/FRONTIER/with_neigbourghs_and_frontier_V2.csv')
base_data = pd.read_csv('./nn_frontier_V3_provisoire_10000.csv')
base_data.index = base_data['id']
# data = pd.read_csv('./with_neigbourghs_and_frontier_V1.csv')
# iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
# iris_df['species'] = pd.Series(iris.target).map(dict(zip(range(3),iris.target_names)))
# sns.pairplot(iris_df, hue='species');
data_target = base_data['class']
# data_target = [x if x != 'frontier' else 3 for x in data_target]
data_target = [int(x) if x == "0" or x == "1" else 3 for x in data_target]


# iris_df['species'] = pd.Series(iris.target).map(dict(zip(range(3),iris.target_names)))
# sns.pairplot(data, hue='class')


# In[6]:


import umap


# In[7]:


reducer = umap.UMAP()


# In[9]:


data = base_data.drop(['Unnamed: 0','id', 'pred_class_0', 'pred_class_1', 'class','good_predict','frontier','nearest','value_0','value_1'],1)


result, sigmas, rhos = umap.umap_.fuzzy_simplicial_set(data.to_numpy(np.float32), 5, np.random.RandomState(42), 'euclidean')



def tulip_creator(matrix, data, base_data, name):
    with open(str(name)+'.tlp', 'a') as the_file:
        the_file.write('(tlp "2.3"\n')
        the_file.write('(nb_nodes '+str(data.to_numpy().shape[0]) +')\n')
        edges = ''
        nodes = '(nodes '
        weigth = '(property 0 double "viewMetric"\n'
        weigth+= '\t(default "0" "0" )\n'
        label = '(property  0 string "viewLabel"\n'
        label += '\t(default "" "")\n'
        class_col = '(property  0 color "viewColor"\n'
        class_col += '\t(default "(235,0,23,255)" "(0,0,0,120)")\n'
        egdes_id = 0
        for i in range(data.to_numpy().shape[0]):
            for nn in matrix[i]:
                value_indice = 0
                nodes += str(i)+" "
                label += '(node '+str(i)+' "'+str(data.iloc[i].name)+'")\n'
                if base_data.loc[data.iloc[i].name]['class']=='frontier':
                    class_col += '(node '+str(i)+' "(200,0,0,255)")\n'
                elif int(base_data.loc[data.iloc[i].name]['class'])==1:
                    class_col += '(node '+str(i)+' "(0,200,0,255)")\n'
                else:
                    class_col += '(node '+str(i)+' "(0,0,200,255)")\n'
                for k in nn.indices:
                    value = nn.data[value_indice]
                    value_indice += 1
                    if i < k:
                        egdes_id += 1
                        edges += '(edge '+str(egdes_id)+' '+str(i)+' '+str(k)+')\n'
                        weigth += '\t(edge '+str(egdes_id)+' "'+str(value)+'" )\n'
        the_file.write(nodes+')\n')
        the_file.write('(nb_edges '+str(egdes_id)+')\n')
        the_file.write(edges)
        the_file.write(weigth+'\t)\n')
        the_file.write(label+'\t)\n')
        the_file.write(class_col+'\t)\n')
        the_file.write(')')


# In[22]:


tulip_creator(result, data, base_data, 'graph_V2')
