debug=False

# data
KG_path = 'data/knowledge_graph'
drop_labels = ['Biological_sample', 'Subject', 'Chromosome', 'Publication', 'GWAS_study', 'Project']

# model 
in_channels = 1024
hidden_channels = 256
n_layers = 3
n_heads = 4
dropout = 0.1
residual = True
margin = 1

# training
batch_size = 32
max_epochs = 50
num_sampled_neighbors = 10 
num_gpu = 2
# learning_rate = 1e-3

# experiment management
save_as = "GAT"