import pandas as pd
import torch
import numpy as np
from torch_geometric.data import HeteroData


def build_heterogeneous_graph(df_path, drug_cols, output_path):
    df = pd.read_csv(df_path, encoding='gb2312', low_memory=False)

    symptoms_list = ['Fever', 'Cough', 'Fatigue', 'Poor appetite', 'Pain']
    diseases_list = ['Pneumonia', 'Upper respiratory tract infection', 'Hypotension', 'Sepsis', 'Bone marrow suppression', 'Diarrhea', 'Anemia', 'Leukemia']

    all_feature_cols = [col for col in df.columns if col not in ['Patient_id', 'label']]
    clinical_cols = [col for col in all_feature_cols if col not in drug_cols]
    hetero_data = HeteroData()

    # Patient nodes
    patient_features = []
    patient_labels = []
    for idx, row in df.iterrows():
        patient_feature = row[all_feature_cols].values.astype(np.float32)
        patient_features.append(patient_feature)
        patient_labels.append(int(row['label']))
    hetero_data['patient'].x = torch.tensor(np.array(patient_features), dtype=torch.float)
    hetero_data['patient'].y = torch.tensor(patient_labels, dtype=torch.long)
    drug_features = torch.eye(len(drug_cols), dtype=torch.float)
    hetero_data['drug'].x = drug_features

    # Symptoms nodes
    symptom_features = torch.eye(len(symptoms_list), dtype=torch.float)
    hetero_data['symptom'].x = symptom_features

    # Diseases nodes
    disease_features = torch.eye(len(diseases_list), dtype=torch.float)
    hetero_data['disease'].x = disease_features

    print("CTD data")
    gene_file_path = 'CTD.csv'
    gene_df = pd.read_csv(gene_file_path, encoding='gb2312', low_memory=False)
    genes_list = gene_df.columns.tolist()[1:]

    # Gene nodes
    gene_features = torch.eye(len(genes_list), dtype=torch.float)
    hetero_data['gene'].x = gene_features
    print(f"Number of gene nodes: {len(genes_list)}")

    print("Read phenotype-gene relationship data")
    phenotype_gene_path = 'CTD\\phenotype\\CTD-phenotype.csv'
    phenotype_gene_df = pd.read_csv(phenotype_gene_path, encoding='gb2312', low_memory=False)
    phenotype_ids = phenotype_gene_df.iloc[:, 0].tolist()
    genes_in_phenotype_df = phenotype_gene_df.columns[1:]

    # Phenotype nodes
    phenotype_features = torch.eye(len(phenotype_ids), dtype=torch.float)
    hetero_data['phenotype'].x = phenotype_features
    print(f"Number of phenotype nodes: {len(phenotype_ids)}")

    # edge indexs
    patient_to_drug_edges = []
    drug_to_patient_edges = []

    patient_to_symptom_edges = []
    symptom_to_patient_edges = []

    patient_to_disease_edges = []
    disease_to_patient_edges = []

    drug_to_gene_edges = []
    gene_to_drug_edges = []

    gene_to_phenotype_edges = []
    phenotype_to_gene_edges = []

    for patient_idx, row in df.iterrows():
        for drug_idx, drug in enumerate(drug_cols):
            if row[drug] == 1:
                patient_to_drug_edges.append([patient_idx, drug_idx])
                drug_to_patient_edges.append([drug_idx, patient_idx])

    for patient_idx, row in df.iterrows():
        for symptom_idx, symptom in enumerate(symptoms_list):
            if symptom in df.columns and row[symptom] == 1:
                patient_to_symptom_edges.append([patient_idx, symptom_idx])
                symptom_to_patient_edges.append([symptom_idx, patient_idx])

    for patient_idx, row in df.iterrows():
        for disease_idx, disease in enumerate(diseases_list):
            if disease in df.columns and row[disease] == 1:
                patient_to_disease_edges.append([patient_idx, disease_idx])
                disease_to_patient_edges.append([disease_idx, patient_idx])

    for _, row in gene_df.iterrows():
        drug_name = row['Chemical Names']
        if drug_name in drug_cols:
            drug_idx = drug_cols.index(drug_name)
            for gene_idx, gene in enumerate(genes_list):
                if row[gene] == 1:
                    drug_to_gene_edges.append([drug_idx, gene_idx])
                    gene_to_drug_edges.append([gene_idx, drug_idx])

    for _, row in phenotype_gene_df.iterrows():
        phenotype_id = row[0]
        phenotype_idx = phenotype_ids.index(phenotype_id)
        for gene in genes_in_phenotype_df:
            if gene in genes_list and row[gene] == 1:
                gene_idx = genes_list.index(gene)
                gene_to_phenotype_edges.append([gene_idx, phenotype_idx])
                phenotype_to_gene_edges.append([phenotype_idx, gene_idx])

    if patient_to_drug_edges:
        hetero_data['patient', 'takes', 'drug'].edge_index = torch.tensor(
            patient_to_drug_edges, dtype=torch.long).t().contiguous()

    if drug_to_patient_edges:
        hetero_data['drug', 'taken_by', 'patient'].edge_index = torch.tensor(
            drug_to_patient_edges, dtype=torch.long).t().contiguous()

    if patient_to_symptom_edges:
        hetero_data['patient', 'has', 'symptom'].edge_index = torch.tensor(
            patient_to_symptom_edges, dtype=torch.long).t().contiguous()

    if symptom_to_patient_edges:
        hetero_data['symptom', 'had_by', 'patient'].edge_index = torch.tensor(
            symptom_to_patient_edges, dtype=torch.long).t().contiguous()

    if patient_to_disease_edges:
        hetero_data['patient', 'has', 'disease'].edge_index = torch.tensor(
            patient_to_disease_edges, dtype=torch.long).t().contiguous()

    if disease_to_patient_edges:
        hetero_data['disease', 'had_by', 'patient'].edge_index = torch.tensor(
            disease_to_patient_edges, dtype=torch.long).t().contiguous()

    if drug_to_gene_edges:
        hetero_data['drug', 'interacts_with', 'gene'].edge_index = torch.tensor(
            drug_to_gene_edges, dtype=torch.long).t().contiguous()

    if gene_to_drug_edges:
        hetero_data['gene', 'interacted_by', 'drug'].edge_index = torch.tensor(
            gene_to_drug_edges, dtype=torch.long).t().contiguous()

    if gene_to_phenotype_edges:
        hetero_data['gene', 'associated_with', 'phenotype'].edge_index = torch.tensor(
            gene_to_phenotype_edges, dtype=torch.long).t().contiguous()

    if phenotype_to_gene_edges:
        hetero_data['phenotype', 'associated_by', 'gene'].edge_index = torch.tensor(
            phenotype_to_gene_edges, dtype=torch.long).t().contiguous()

    torch.save(hetero_data, output_path)
    print(f"The heterogeneous network has been saved to: {output_path}")

    # node and edge statistics
    print(f"Number of patient nodes: {hetero_data['patient'].num_nodes}")
    print(f"Number of drug nodes: {hetero_data['drug'].num_nodes}")
    print(f"Number of symptom nodes: {hetero_data['symptom'].num_nodes}")
    print(f"Number of disease nodes: {hetero_data['disease'].num_nodes}")
    print(f"Number of gene nodes: {hetero_data['gene'].num_nodes}")
    print(f"Number of phenotype nodes: {hetero_data['phenotype'].num_nodes}")
    for edge_type in hetero_data.edge_types:
        edge_index = hetero_data[edge_type].edge_index
        if edge_index is not None:
            src_type, _, dst_type = edge_type
            print(f"{edge_type}Number of edge nodes: {edge_index.shape[1]}")
    return hetero_data

if __name__ == "__main__":
    CONFIG = {
        'train_path': 'ADR-Train.csv',
        'valid_path': 'ADR-Valid.csv',
    }
    drug_cols = ['Penicillins', 'Cephalosporins', 'Aminoglycosides', 'Macrolides', 'Antitumor antibiotics',
                 'Other antibiotics', 'Ibuprofen', 'Dexamethasone', 'Ambroxol', 'Lidocaine', 'Cytarabine', 'Ribavirin',
                 'Granisetron', 'Cyclophosphamide', 'Amino acids', 'Heparin sodium', 'Mesna',
                 'Carboxymethyl sulfonate sodium', 'Methotrexate', 'Furosemide', 'Budesonide',
                 'Chloral hydrate', 'Prednisone', 'Mannitol', 'Vincristine', 'Ipratropium bromide', 'Omeprazole sodium',
                 'Inflammation suppressant', 'Creatine sodium', 'Cimetidine', 'Salbutamol', 'Glycerin enema',
                 'Recombinant human granulocyte colony-stimulating factor', 'Mint', 'Montmorillonite', 'Kangfuxin',
                 'Cyclophosphamide', 'Mercaptopurine', 'Camphor', 'Calcium carbonate', 'Loratadine',
                 'Triple live bacteria of Bifidobacterium and Streptococcus thermophilus']

    print("Construct a heterogeneous training set network")
    train_hetero_data = build_heterogeneous_graph(
        CONFIG['train_path'],
        drug_cols,
        'train_graph-Hetero.pt'
    )
    print("\n Constructing a validation set heterogeneous network")
    valid_hetero_data = build_heterogeneous_graph(
        CONFIG['valid_path'],
        drug_cols,
        'valid_graph-Hetero.pt'
    )