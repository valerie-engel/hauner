debug=False

# data
KG_path = 'data/knowledge_graph'
# drop_labels = ['Subject', 'Chromosome', 'Publication', 'GWAS_study', 'Project'] # 'Biological_sample', 
select_labels = ['Disease', 'Phenotype', 'Gene', 'Protein']
drop_selected_labels = False
undirected=False
embedding=None
drop_unembedded=True

# model 
in_channels = 128
hidden_channels = 128
out_channels = 128
n_layers = 3
n_heads = 4
dropout = 0.1
residual = False
margin = 1

# training
batch_size = 256
max_epochs = 20
num_sampled_neighbors = 10 
# learning_rate = 1e-3

# experiment management
save_as = "small_GAT"

if debug: 
    #data
    KG_path = 'data/knowledge_graph' #tiny_
    # drop_labels = ['Biological_sample', 'Subject', 'Chromosome', 'Publication', 'GWAS_study', 'Project'] #None # 
    select_labels = ['Disease', 'Phenotype', 'Gene', 'Protein']
    drop_selected_labels = False
    # embedding='fastrp'
    drop_unembedded=True

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
    max_epochs = 3
    num_sampled_neighbors = 2 

    # experiment management
    save_as = "test"