debug=False

# data
KG_path = 'data/knowledge_graph'
drop_labels = ['Biological_sample', 'Subject', 'Chromosome', 'Publication', 'GWAS_study', 'Project']

# model 
in_channels = 256
hidden_channels = 256
out_channels = 256
n_layers = 3
n_heads = 4
dropout = 0.1
residual = False
margin = 1

# training
batch_size = 16
max_epochs = 20
num_sampled_neighbors = 10 
# learning_rate = 1e-3

# experiment management
save_as = "GAT_embed_by_type"

if debug: 
    #data
    KG_path = 'data/tiny_knowledge_graph' #tiny_
    drop_labels = None #['Biological_sample', 'Subject', 'Chromosome', 'Publication', 'GWAS_study', 'Project'] #None

    in_channels = 8
    hidden_channels = 16
    out_channels = 32
    n_layers = 2
    n_heads = 2
    dropout = 0.1
    residual = False
    margin = 1

    # training
    batch_size = 16
    max_epochs = 2
    num_sampled_neighbors = 2 

    # experiment management
    save_as = "test"