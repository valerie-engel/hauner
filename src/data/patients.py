import torch
from torch_geometric.data import Dataset # Data, 
from torch_geometric.utils import k_hop_subgraph
import sys

from .knowledge_graph import HaunerGraph

class PatientData(HaunerGraph):
    def __init__(self, cfg, discard_undiagnosed=False):
        select_labels_with_patients = cfg.select_labels.copy()
        select_labels_with_patients.append('Biological_sample')
        super().__init__(
            path=cfg.KG_path, 
            device='cpu', 
            undirected=cfg.undirected, 
            embedding=cfg.embedding, 
            select_labels=select_labels_with_patients, 
            drop_selected_labels=cfg.drop_selected_labels,
            drop_unembedded=cfg.drop_unembedded
            )
        self.disease_id = self.ids_of_labels(['Disease'])
        self.patients = self.nodes_of_labels(['Biological_sample'])
        print(f"Found {len(self.patients)} biological samples")
        self.diagnoses = [self.get_diagnosis(patient) for patient in self.patients]
        if discard_undiagnosed:
            self.discard_undiagnosed()

    def get_diagnosis(self, patient):
        neighbors = self.edge_index[1][self.edge_index[0] == patient]
        # print(f"neighbors: {neighbors}")
        diagnoses_ids = neighbors[self.y[neighbors] == self.disease_id]
        # print(f"diagnoses_ids: {diagnoses_ids}")
        return diagnoses_ids

    def discard_undiagnosed(self):
        print("Discarding undiagnosed patients")
        diagnosed = [i for i, diagnosis in enumerate(self.diagnoses) if diagnosis.numel() > 0]
        self.patients = self.patients[diagnosed]
        self.diagnoses = [self.diagnoses[i] for i in diagnosed]
        print(f"Remaining patients: {len(self.diagnoses)}")


# # DataLoaders?
# class PatientData(Dataset): 
#     def __init__(self, cfg, num_hops, load_from_path=None, discard_undiagnosed=False):
#         super().__init__()
#         self.KG = self.load_KG(cfg)
#         self.num_hops = num_hops
#         self.disease_id = self.KG.ids_of_labels(['Disease'])
#         if load_from_path:
#             self.patients, self.diagnoses = self.load_patients(load_from_path)
#         else:
#             self.patients, self.diagnoses = self.construct_dataset()
#         self.print_stats()
#         if discard_undiagnosed: 
#             self.discard_undiagnosed()
#         # self.KG = None # save space ALTERNATIVE: store KG and only subsets of node_ids per Patient, query KG in get()


#     def len(self):
#         return len(self.diagnoses)


#     def get(self, idx):
#         patient = (self.KG.x[self.patients[idx]], self.KG.y[self.patients[idx]])
#         diagnosis = self.KG.x[self.diagnoses[idx]]
#         return patient, diagnosis 


#     def load_KG(self, cfg):# DO THIS VIA STORED ARGS! NOT RIGHT CFG
#         select_labels_with_patients = cfg.select_labels.copy()
#         select_labels_with_patients.append('Biological_sample')

#         KG = HaunerGraph(
#             path=cfg.KG_path, 
#             device='cpu',
#             undirected=cfg.undirected, 
#             embedding=cfg.embedding, 
#             select_labels=select_labels_with_patients, 
#             drop_selected_labels=cfg.drop_selected_labels,
#             drop_unembedded=cfg.drop_unembedded
#             ) 
#         return KG 


    # def construct_dataset(self):
    #     patient_nodes = self.KG.nodes_of_labels(['Biological_sample'])
    #     print(f"Found {len(patient_nodes)} biological samples")

    #     patient = patient_nodes[0]
    #     d = self.get_diagnosis(patient)
    #     neighbors, _, _, _ = k_hop_subgraph([patient], 1, self.KG.edge_index)   #, flow="target_to_source"
    #     print(f"k hop neighbors: {neighbors}, len: {len(neighbors)}")
    #     p = self.k_hop_neighborhood(center_nodes=[patient], ignore_nodes=d, discard_center_nodes=True)
    #     sys.exit()
    #     diagnoses = [self.get_diagnosis(patient) for patient in patient_nodes]  # does this return empty when nothing found?
    #     patients = [self.k_hop_neighborhood(center_nodes=[patient], ignore_nodes=diagnosis, discard_center_nodes=True) for patient, diagnosis in zip(patient_nodes, diagnoses)]   # keep mappings?
    #     return patients, diagnoses


    # def k_hop_neighborhood(self, center_nodes, ignore_nodes, discard_center_nodes=False):
    #     # ignore ignore_nodes
    #     src, dst = self.KG.edge_index
    #     mask = (~torch.isin(src, ignore_nodes)) & (~torch.isin(dst, ignore_nodes))
    #     filtered_edge_index = self.KG.edge_index[:, mask]
    #     # print(filtered_edge_index.shape, self.KG.edge_index.shape)

    #     # compute which nodes to keep
    #     neighbors, _, _, _ = k_hop_subgraph(center_nodes, self.num_hops, filtered_edge_index) #, flow="target_to_source"
    #     print(f"k hop neighbors: {neighbors}, len: {len(neighbors)}")

    #     if discard_center_nodes:
    #         print(center_nodes)
    #         neighbors = torch.tensor(list(set(neighbors)-set(center_nodes)))
    #     print(f"k hop neighbors: {neighbors}")
    #     return neighbors

    # # def get_patient_subgraph(self, center_nodes, diagnosis):
    # #     # ignore any information coming from diagnosis
    # #     src, dst = self.KG.edge_index
    # #     mask = (~torch.isin(src, diagnosis)) & (~torch.isin(dst, diagnosis))
    # #     filtered_edge_index = self.KG.edge_index[:, mask]
    # #     filtered_edge_label = self.KG.edge_label[mask]

    # #     # compute which nodes to keep
    # #     subset, edge_index, mapping, edge_mask = k_hop_subgraph(center_nodes, self.num_hops, filtered_edge_index, relabel_nodes=True)
    # #     # or don't construct subgraphs and only index into main graph? and don't relabel nodes?
    # #     subgraph = Data(
    # #         # x=self.KG.x[subset],
    # #         edge_index=edge_index,
    # #         edge_label=filtered_edge_label[edge_mask],    
    # #         y=self.KG.y[subset]   
    # #     )
    # #     # print(f"{k} hop subgraph contains {subgraph.num_nodes} nodes and {subgraph.num_edges} edges")
    # #     return subgraph #, mapping



    # def print_stats(self):
    #     neighborhood_sizes = [len(patient) for patient in self.patients]
    #     print(f"Patient neighborhood sizes:")
    #     print(f"smallest: {min(neighborhood_sizes)}, average: {sum(neighborhood_sizes)/len(neighborhood_sizes)}, largest: {max(neighborhood_sizes)}")
    # #     subgraph_sizes = [[subgraph.num_nodes, subgraph.num_edges] for subgraph in self.patients]
    # #     print(f"Patient subgraph sizes:")
    # #     node_counts = subgraph_sizes[:][0]
    # #     print(f"node counts:\n smallest: {min(node_counts)}, average: {sum(node_counts)/len(node_counts)}, largest: {max(node_counts)}")
    # #     edge_counts = subgraph_sizes[:][1]
    # #     print(f"edge counts:\n smallest: {min(edge_counts)}, average: {sum(edge_counts)/len(edge_counts)}, largest: {max(edge_counts)}")

    #     num_diseases = [len(diagnosis) for diagnosis in self.diagnoses]
    #     print(f"Patients without diagnosis: {sum(num_diseases==0)}")
    #     print(f"Patients with multiple diseases: {sum(num_diseases>1)}")
    #     print(f"Maximum number of diseases: {max(num_diseases)}")


    # def save_patients(path):
    #     path = os.path.join(path, f'{self.num_hops}_hops')
    #     if not os.path.exists(path):
    #         os.makedirs(path)

    #     torch.save(self.patients, os.path.join(path, 'patients.pt'))
    #     torch.save(self.diagnoses, os.path.join(path, 'diagnoses.pt'))


    # def load_patients(path):
    #     path = os.path.join(path, f'{self.num_hops}_hops')
    #     patients = torch.load(os.path.join(path, 'patients.pt'))
    #     diagnoses = torch.load(os.path.join(path, 'diagnoses.pt'))
    #     return patients, diagnoses
        

