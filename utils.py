import os
import numpy as np
import random
import gc
import json
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


def partition_multi_task_dataset(task_name_list, tokenizer, alpha=0.5, train_samples_per_client=None, 
                              test_samples_per_client=None, seed=42):
    
    np.random.seed(seed)
    random.seed(seed)
    
    num_clients = len(task_name_list)
    print(f"Creating datasets for {num_clients} clients with tasks: {task_name_list}")
    
    unique_tasks = set(task_name_list)
    task_datasets = {}
    task_info = {}
    
    task_available_indices = {}
    
    for task_name in unique_tasks:
        raw_datasets = load_dataset("glue", task_name)
        
        num_labels = len(set(raw_datasets["train"]["label"]))

        if task_name == "sst2":
            sentence1_key, sentence2_key = "sentence", None
        elif task_name == "qnli":
            sentence1_key, sentence2_key = "question", "sentence"
        elif task_name == "mnli":
            sentence1_key, sentence2_key = "premise", "hypothesis"
        elif task_name == "qqp":
            sentence1_key, sentence2_key = "question1", "question2"
        elif task_name == "rte" or task_name == "mrpc":
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            raise ValueError(f"Task {task_name} not supported")
        
        validation_key = "validation_matched" if task_name == "mnli" else "validation"
        
        task_datasets[task_name] = {
            "raw_datasets": raw_datasets,
            "num_labels": num_labels,
            "sentence1_key": sentence1_key,
            "sentence2_key": sentence2_key,
            "validation_key": validation_key
        }
        
        train_labels = np.array([example['label'] for example in raw_datasets["train"]])
        val_labels = np.array([example['label'] for example in raw_datasets[validation_key]])
        
        task_available_indices[task_name] = {
            "train": [np.where(train_labels == k)[0] for k in range(num_labels)],
            validation_key: [np.where(val_labels == k)[0] for k in range(num_labels)]
        }
        
        if task_name == "mnli" and "validation_mismatched" in raw_datasets:
            mismatched_labels = np.array([example['label'] for example in raw_datasets["validation_mismatched"]])
            task_available_indices[task_name]["validation_mismatched"] = [
                np.where(mismatched_labels == k)[0] for k in range(num_labels)
            ]
        
        for split in task_available_indices[task_name]:
            for k in range(num_labels):
                indices = task_available_indices[task_name][split][k].tolist()
                random.shuffle(indices)
                task_available_indices[task_name][split][k] = np.array(indices)
        
        print(f"Dataset {task_name} has {num_labels} classes")
    
    task_client_counts = {}
    for task in unique_tasks:
        task_client_counts[task] = sum(1 for t in task_name_list if t == task)
    
    def partition_dataset_for_client(client_id, task_name, client_seed, task_client_idx):
        client_random = random.Random(client_seed)
        client_np_random = np.random.RandomState(client_seed)
        
        task_data = task_datasets[task_name]
        raw_datasets = task_data["raw_datasets"]
        num_labels = task_data["num_labels"]
        sentence1_key = task_data["sentence1_key"]
        sentence2_key = task_data["sentence2_key"]
        validation_key = task_data["validation_key"]

        n_classes = num_labels

        proportion = np.ones(n_classes) / n_classes
        
        min_proportion = 0.01 

        small_prop_indices = np.where(proportion < min_proportion)[0]
        
        if len(small_prop_indices) > 0:
            deficit = min_proportion * len(small_prop_indices) - proportion[small_prop_indices].sum()

            large_prop_indices = np.where(proportion > min_proportion)[0]
            
            if len(large_prop_indices) > 0:
                reduction_per_class = deficit / len(large_prop_indices)
                proportion[large_prop_indices] -= reduction_per_class
                proportion[small_prop_indices] = min_proportion

                proportion = proportion / proportion.sum()

        print(f"Client {client_id} ({task_name}) class proportions: {proportion}")

        train_size = train_samples_per_client
        if train_size is None:
            train_size = len(raw_datasets["train"]) // task_client_counts[task_name]
        
        test_size = test_samples_per_client
        if test_size is None:
            test_size = len(raw_datasets[validation_key]) // task_client_counts[task_name]
        
        print(f"Client {client_id} ({task_name}) train samples: {train_size}, test samples: {test_size}")

        min_samples_per_class = 1

        client_train_class_counts = np.round(proportion * train_size).astype(int)
        client_val_class_counts = np.round(proportion * test_size).astype(int)

        for k in range(n_classes):
            if client_train_class_counts[k] < min_samples_per_class:
                client_train_class_counts[k] = min_samples_per_class
            
            if client_val_class_counts[k] < min_samples_per_class:
                client_val_class_counts[k] = min_samples_per_class
        
        train_diff = train_size - client_train_class_counts.sum()
        if train_diff != 0:
            available_train = [len(task_available_indices[task_name]["train"][k]) for k in range(n_classes)]
            
            if train_diff > 0:
                classes_to_adjust = np.argsort(-np.array(available_train))
                
                for idx in classes_to_adjust:
                    if train_diff > 0 and available_train[idx] > 0:
                        add_count = min(train_diff, available_train[idx])
                        client_train_class_counts[idx] += add_count
                        train_diff -= add_count
                    
                    if train_diff == 0:
                        break
            else:
                classes_to_adjust = np.argsort(-client_train_class_counts)
                
                for idx in classes_to_adjust:
                    
                    if train_diff < 0 and client_train_class_counts[idx] > min_samples_per_class:
                        remove_count = min(-train_diff, client_train_class_counts[idx] - min_samples_per_class)
                        client_train_class_counts[idx] -= remove_count
                        train_diff += remove_count
                    
                    if train_diff == 0:
                        break
        
        val_diff = test_size - client_val_class_counts.sum()
        if val_diff != 0:
            available_val = [len(task_available_indices[task_name][validation_key][k]) for k in range(n_classes)]
            
            if val_diff > 0:
                classes_to_adjust = np.argsort(-np.array(available_val))
                
                for idx in classes_to_adjust:
                    if val_diff > 0 and available_val[idx] > 0:
                        add_count = min(val_diff, available_val[idx])
                        client_val_class_counts[idx] += add_count
                        val_diff -= add_count
                    
                    if val_diff == 0:
                        break
            else:
                classes_to_adjust = np.argsort(-client_val_class_counts)
                
                for idx in classes_to_adjust:
                    if val_diff < 0 and client_val_class_counts[idx] > min_samples_per_class:
                        remove_count = min(-val_diff, client_val_class_counts[idx] - min_samples_per_class)
                        client_val_class_counts[idx] -= remove_count
                        val_diff += remove_count
                    
                    if val_diff == 0:
                        break
        
        client_train_indices = []
        client_val_indices = []
        
        for k in range(n_classes):
            if client_train_class_counts[k] > 0:
                train_needed = client_train_class_counts[k]
                available_indices = task_available_indices[task_name]["train"][k]
                
                if train_needed <= len(available_indices):
                    client_train_indices.extend(available_indices[:train_needed].tolist())
                    task_available_indices[task_name]["train"][k] = available_indices[train_needed:]
                else:
                    client_train_indices.extend(available_indices.tolist())
                    
                    additional_needed = train_needed - len(available_indices)
                    if additional_needed > 0:
                        original_indices = np.where(np.array([example['label'] for example in raw_datasets["train"]]) == k)[0]
                        used_indices = set(np.concatenate([client_train_indices] + 
                                            [c_indices for c_idx, c_indices in enumerate(task_available_indices[task_name]["train"]) 
                                            if c_idx != k]))
                        available_pool = [idx for idx in original_indices if idx not in used_indices]
                        
                        if available_pool:
                            additional = client_np_random.choice(
                                available_pool, 
                                min(additional_needed, len(available_pool)), 
                                replace=False
                            ).tolist()
                        else:
                            additional = client_np_random.choice(
                                original_indices, 
                                additional_needed, 
                                replace=True
                            ).tolist()
                        
                        client_train_indices.extend(additional)
                    
                    task_available_indices[task_name]["train"][k] = np.array([])
            
            if client_val_class_counts[k] > 0:
                val_needed = client_val_class_counts[k]
                available_indices = task_available_indices[task_name][validation_key][k]
                
                if val_needed <= len(available_indices):
                    client_val_indices.extend(available_indices[:val_needed].tolist())
                    task_available_indices[task_name][validation_key][k] = available_indices[val_needed:]
                else:
                    client_val_indices.extend(available_indices.tolist())
                    
                    additional_needed = val_needed - len(available_indices)
                    if additional_needed > 0:
                        original_indices = np.where(np.array([example['label'] for example in raw_datasets[validation_key]]) == k)[0]
                        used_indices = set(np.concatenate([client_val_indices] + 
                                            [c_indices for c_idx, c_indices in enumerate(task_available_indices[task_name][validation_key]) 
                                            if c_idx != k]))
                        available_pool = [idx for idx in original_indices if idx not in used_indices]
                        
                        if available_pool:
                            additional = client_np_random.choice(
                                available_pool, 
                                min(additional_needed, len(available_pool)), 
                                replace=False
                            ).tolist()
                        else:
                            additional = client_np_random.choice(
                                original_indices, 
                                additional_needed, 
                                replace=True
                            ).tolist()
                        
                        client_val_indices.extend(additional)
                    
                    task_available_indices[task_name][validation_key][k] = np.array([])
        
        max_length = 128
        def preprocess_function(examples):
            texts = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*texts, padding=False, max_length=max_length, truncation=True)
            
            if "label" in examples:
                result["labels"] = examples["label"]
            return result
        
        client_train_data = raw_datasets["train"].select(client_train_indices)
        client_val_data = raw_datasets[validation_key].select(client_val_indices)
        
        client_train_tokenized = client_train_data.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc=f"Tokenizing train data for client {client_id}",
        )
        
        client_val_tokenized = client_val_data.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets[validation_key].column_names,
            desc=f"Tokenizing validation data for client {client_id}",
        )
        
        client_dict = {
            "train": client_train_tokenized,
            "validation": client_val_tokenized
        }
        
        if task_name == "mnli" and "validation_mismatched" in raw_datasets:
            mismatched_val_indices = []
            
            mismatched_size = test_size
            client_mismatched_class_counts = np.round(proportion * mismatched_size).astype(int)
            
            for k in range(n_classes):
                if client_mismatched_class_counts[k] < min_samples_per_class:
                    client_mismatched_class_counts[k] = min_samples_per_class
            
            diff = mismatched_size - client_mismatched_class_counts.sum()
            if diff != 0:
                if diff > 0:
                    classes_to_adjust = np.argsort(-proportion)
                    for idx in classes_to_adjust:
                        if diff > 0:
                            client_mismatched_class_counts[idx] += 1
                            diff -= 1
                        
                        if diff == 0:
                            break
                else:
                    classes_to_adjust = np.argsort(-client_mismatched_class_counts)
                    for idx in classes_to_adjust:
                        if diff < 0 and client_mismatched_class_counts[idx] > min_samples_per_class:
                            remove_count = min(-diff, client_mismatched_class_counts[idx] - min_samples_per_class)
                            client_mismatched_class_counts[idx] -= remove_count
                            diff += remove_count
                        
                        if diff == 0:
                            break
            
            for k in range(n_classes):
                if client_mismatched_class_counts[k] > 0:
                    mismatched_needed = client_mismatched_class_counts[k]
                    available_indices = task_available_indices[task_name]["validation_mismatched"][k]
                    
                    if mismatched_needed <= len(available_indices):
                        mismatched_val_indices.extend(available_indices[:mismatched_needed].tolist())
                        task_available_indices[task_name]["validation_mismatched"][k] = available_indices[mismatched_needed:]
                    else:
                        mismatched_val_indices.extend(available_indices.tolist())
                        
                        additional_needed = mismatched_needed - len(available_indices)
                        if additional_needed > 0:
                            original_indices = np.where(np.array([example['label'] for example in raw_datasets["validation_mismatched"]]) == k)[0]
                            used_indices = set(np.concatenate([mismatched_val_indices] + 
                                                [c_indices for c_idx, c_indices in enumerate(task_available_indices[task_name]["validation_mismatched"]) 
                                                if c_idx != k]))
                            available_pool = [idx for idx in original_indices if idx not in used_indices]
                            
                            if available_pool:
                                additional = client_np_random.choice(
                                    available_pool, 
                                    min(additional_needed, len(available_pool)), 
                                    replace=False
                                ).tolist()
                            else:
                                additional = client_np_random.choice(
                                    original_indices, 
                                    additional_needed, 
                                    replace=True
                                ).tolist()
                            
                            mismatched_val_indices.extend(additional)
                        
                        task_available_indices[task_name]["validation_mismatched"][k] = np.array([])
            
            client_mismatched_data = raw_datasets["validation_mismatched"].select(mismatched_val_indices)
            client_mismatched_tokenized = client_mismatched_data.map(
                preprocess_function,
                batched=True,
                remove_columns=raw_datasets["validation_mismatched"].column_names,
                desc=f"Tokenizing mismatched validation for client {client_id}",
            )
            client_dict["validation_mismatched"] = client_mismatched_tokenized
        
        return DatasetDict(client_dict)
    
    client_datasets = {}
    
    task_client_indices = {task: 0 for task in unique_tasks}
    
    for client_id in range(num_clients):
        task_name = task_name_list[client_id]
        
        client_seed = seed * 10000 + client_id
        
        task_client_idx = task_client_indices[task_name]
        task_client_indices[task_name] += 1
        
        client_datasets[client_id] = partition_dataset_for_client(
            client_id, task_name, client_seed, task_client_idx
        )
        
        task_info[client_id] = {
            "task_name": task_name,
            "num_labels": task_datasets[task_name]["num_labels"]
        }
        
        print(f"\nClient {client_id} ({task_name}) dataset sizes:")
        print(f"  Training: {len(client_datasets[client_id]['train'])}")
        print(f"  Validation: {len(client_datasets[client_id]['validation'])}")
        if task_name == "mnli" and "validation_mismatched" in client_datasets[client_id]:
            print(f"  Validation Mismatched: {len(client_datasets[client_id]['validation_mismatched'])}")
        
        for split in client_datasets[client_id]:
            labels = [example["labels"] for example in client_datasets[client_id][split]]
            unique, counts = np.unique(labels, return_counts=True)
            class_counts = dict(zip(unique, counts))
            print(f"  Class distribution in {split}: {class_counts}")
    
    return client_datasets, task_info



def load_B_only(proj_type, client_id, round_id, base_dir="./"):
    file_path = os.path.join(base_dir, f"{proj_type}_lora_B_client_{client_id}_{round_id}.npy")
    B = np.load(file_path).astype(np.float32)
    gc.collect()
    return B


def calculate_B_similarity_matrix(client_ids, proj_type, round_id, base_dir="./"):

    n_clients = len(client_ids)
    distance_matrix_B = np.zeros((n_clients, n_clients))
    
    client_pairs = [(i, j) for i in range(n_clients) for j in range(i+1, n_clients)]
    
    for (i, j) in tqdm(client_pairs, desc=f"Computing {proj_type} B matrix similarities"):
        client_i = client_ids[i]
        client_j = client_ids[j]
        
        try:
            B1 = load_B_only(proj_type, client_i, round_id, base_dir)
            B2 = load_B_only(proj_type, client_j, round_id, base_dir)
        except FileNotFoundError as e:
            print(f"Error loading matrices: {e}")
            print(f"Skipping client pair ({client_i}, {client_j})")
            continue
        
        B_similarities = []
        start_layer = min(0, B1.shape[0]-1) 
        
        for layer in range(start_layer, B1.shape[0]):
            b1_flat = B1[layer].flatten()
            b2_flat = B2[layer].flatten()
            
            dot_product_B = np.dot(b1_flat, b2_flat)
            norm_B1 = np.linalg.norm(b1_flat)
            norm_B2 = np.linalg.norm(b2_flat)
            cos_sim_B = dot_product_B / (norm_B1 * norm_B2 + 1e-8)
            
            B_similarities.append(1 - cos_sim_B)
            
            del b1_flat, b2_flat
        
        avg_distance_B = np.mean(B_similarities)
        
        distance_matrix_B[i, j] = distance_matrix_B[j, i] = avg_distance_B
        
        del B1, B2
        gc.collect()
    
    return distance_matrix_B


def visualize_clustering(combined_matrix, client_ids, labels, output_dir, round_idx):
    os.makedirs(output_dir, exist_ok=True)
    
    condensed_dist = squareform(combined_matrix)
    Z = linkage(condensed_dist, method='average')
    plt.figure(figsize=(12, 8))
    dendrogram(Z, labels=[str(cid) for cid in client_ids], leaf_rotation=90)
    plt.title('Client Clustering for LoRA Groups')
    plt.xlabel('Client')
    plt.ylabel('Distance (1 - Cosine Similarity)')
    plt.savefig(os.path.join(output_dir, f"lora_clustering_round_{round_idx}.png"))
    plt.close()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(combined_matrix, cmap='viridis')
    plt.colorbar(label='Distance (1 - Cosine Similarity)')
    plt.xticks(range(len(client_ids)), [str(cid) for cid in client_ids], rotation=45)
    plt.yticks(range(len(client_ids)), [str(cid) for cid in client_ids])
    plt.title('B Matrix Distance Between Clients')
    plt.savefig(os.path.join(output_dir, f"lora_distance_matrix_round_{round_idx}.png"))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(range(len(client_ids)), [0] * len(client_ids), c=labels, cmap='viridis', 
                         s=100, marker='o')
    plt.colorbar(scatter, label='Cluster ID')
    plt.xticks(range(len(client_ids)), [str(cid) for cid in client_ids])
    plt.yticks([])
    plt.title('Client Cluster Assignments')
    plt.savefig(os.path.join(output_dir, f"lora_clusters_round_{round_idx}.png"))
    plt.close()


def compute_lora_client_map(clients, round_idx, personal_dir="./", max_clusters=10):
    print("Computing LoRA client map based on B matrix similarity...")
    
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    
    proj_types = ["query", "value"]
    
    client_ids = [client.client_id for client in clients]
    n_clients = len(client_ids)
    
    log_file = os.path.join(personal_dir, "training_log.txt")
    with open(log_file, "a") as f:
        f.write("\nStarting LoRA client mapping calculation...\n")
    
    all_distance_matrices = {}
    
    param_dir = os.path.join(personal_dir, "lora_params")
    
    for proj_type in proj_types:
        with open(log_file, "a") as f:
            f.write(f"Calculating {proj_type} B matrix similarities...\n")
        
        print(param_dir)
        dist_matrix_B = calculate_B_similarity_matrix(client_ids, proj_type, round_idx, base_dir=param_dir)
        all_distance_matrices[proj_type] = dist_matrix_B
        
        np.save(os.path.join(personal_dir, f"{proj_type}_distance_matrix_round_{round_idx}.npy"), dist_matrix_B)
    
    with open(log_file, "a") as f:
        f.write("Combining distance matrices...\n")
    
    combined_matrix = np.zeros_like(all_distance_matrices[proj_types[0]])
    for proj_type in proj_types:
        combined_matrix += all_distance_matrices[proj_type]
    combined_matrix /= len(proj_types)
    
    np.save(os.path.join(personal_dir, f"combined_distance_matrix_round_{round_idx}.npy"), combined_matrix)
    
    with open(log_file, "a") as f:
        f.write(f"Evaluating optimal cluster number from 1 to {max_clusters}...\n")
    
    silhouette_scores = []
    davies_bouldin_scores = []
    cluster_labels_list = []
    
    from sklearn.manifold import MDS
    embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    embedded_coords = embedding.fit_transform(combined_matrix)
    
    plt.figure(figsize=(12, 6))
    
    for n_clusters in range(2, max_clusters + 1):
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        
        labels = model.fit_predict(combined_matrix)
        cluster_labels_list.append(labels)
        
        try:
            sil_score = silhouette_score(
                combined_matrix, 
                labels, 
                metric='precomputed'
            )
            silhouette_scores.append(sil_score)
            
            db_score = davies_bouldin_score(embedded_coords, labels)
            davies_bouldin_scores.append(db_score)
            
            with open(log_file, "a") as f:
                f.write(f"  Clusters={n_clusters}, Silhouette Score={sil_score:.4f}, Davies-Bouldin Score={db_score:.4f}\n")
                
            print(f"Clusters={n_clusters}, Silhouette Score={sil_score:.4f}, Davies-Bouldin Score={db_score:.4f}")
        except Exception as e:
            with open(log_file, "a") as f:
                f.write(f"  Error with n_clusters={n_clusters}: {str(e)}\n")
            silhouette_scores.append(-1) 
            davies_bouldin_scores.append(float('inf'))
    
    plt.plot(range(2, max_clusters + 1), silhouette_scores, 'b-o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.grid(True)
    plt.savefig(os.path.join(personal_dir, f"silhouette_scores_round_{round_idx}.png"))
    plt.close()
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(2, max_clusters + 1), davies_bouldin_scores, 'r-o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Davies-Bouldin Score (lower is better)')
    plt.title('Davies-Bouldin Score vs. Number of Clusters')
    plt.grid(True)
    plt.savefig(os.path.join(personal_dir, f"davies_bouldin_scores_round_{round_idx}.png"))
    plt.close()
    
    if max(silhouette_scores) - min(silhouette_scores) > 0:
        norm_silhouette = [(s - min(silhouette_scores)) / (max(silhouette_scores) - min(silhouette_scores)) 
                          for s in silhouette_scores]
    else:
        norm_silhouette = [1.0 for _ in silhouette_scores]
    
    if max(davies_bouldin_scores) - min(davies_bouldin_scores) > 0:
        norm_davies = [1 - ((s - min(davies_bouldin_scores)) / (max(davies_bouldin_scores) - min(davies_bouldin_scores)))
                      for s in davies_bouldin_scores]
    else:
        norm_davies = [1.0 for _ in davies_bouldin_scores]
    
    combined_scores = [s for s, d in zip(norm_silhouette, norm_davies)]
    
    best_idx = combined_scores.index(max(combined_scores))
    optimal_n_clusters = best_idx + 2  
    
    optimal_labels = cluster_labels_list[best_idx]
    
    with open(log_file, "a") as f:
        f.write(f"Optimal number of clusters determined: {optimal_n_clusters}\n")
    print(f"Optimal number of clusters: {optimal_n_clusters}")
    
    lora_client_map = {}
    for cluster_id in range(optimal_n_clusters):
        client_indices = [i for i, label in enumerate(optimal_labels) if label == cluster_id]
        lora_client_map[cluster_id] = client_indices
    
    print("\nClustering Results:")
    with open(log_file, "a") as f:
        f.write("\nClustering Results:\n")
        for cluster_id, cluster_clients in lora_client_map.items():
            client_names = [client_ids[i] for i in cluster_clients]
            cluster_info = f"Cluster {cluster_id} (LoRA {cluster_id}): {', '.join(map(str, client_names))}"
            print(cluster_info)
            f.write(cluster_info + "\n")
    
    cluster_file = os.path.join(personal_dir, f"lora_client_map_round_{round_idx}.json")
    with open(cluster_file, 'w') as f:
        json.dump(lora_client_map, f, indent=2)
    
    with open(os.path.join(personal_dir, f"optimal_n_clusters_round_{round_idx}.json"), 'w') as f:
        json.dump({"optimal_n_clusters": optimal_n_clusters}, f, indent=2)
    
    with open(log_file, "a") as f:
        f.write("Generating visualization plots...\n")
    
    visualize_clustering(combined_matrix, client_ids, optimal_labels, personal_dir, round_idx)
    
    plt.figure(figsize=(10, 8))
    for cluster_id in range(optimal_n_clusters):
        cluster_points = [i for i, label in enumerate(optimal_labels) if label == cluster_id]
        plt.scatter(
            embedded_coords[cluster_points, 0],
            embedded_coords[cluster_points, 1],
            label=f'Cluster {cluster_id}',
            s=100
        )
    
    for i, (x, y) in enumerate(embedded_coords):
        plt.annotate(str(client_ids[i]), (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.title(f'Client Clusters in 2D Space (n_clusters={optimal_n_clusters})')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig(os.path.join(personal_dir, f"cluster_embedding_round_{round_idx}.png"))
    plt.close()
    
    with open(log_file, "a") as f:
        f.write("LoRA client mapping calculation completed.\n")
    
    return lora_client_map, optimal_n_clusters


