#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import json
import datetime
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from utils import partition_multi_task_dataset, compute_lora_client_map
from client import Client, WarmupClient
from server import Server


def train_federated(
    dummy,
    clients,
    server,
    global_rounds,
    local_epochs,
    output_dir,
    lr=3e-4,
    round_warmup=1,
    max_clusters=10,
    task_info=None,
    client_datasets=None,
    batch_size=128
):
    personal_dir = os.path.join(output_dir, "proposed_m2")
    os.makedirs(personal_dir, exist_ok=True)
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = os.path.join(personal_dir, "training_log.txt")
    with open(log_file, "w") as f:
        f.write(f"[{current_time}] Starting Federated Training with Dummy Client\n")
        f.write(f"Total Rounds: {global_rounds}, Local Epochs: {local_epochs}, Warmup Rounds: {round_warmup}\n")
        f.write("-" * 50 + "\n")
    
    warmup_clients = clients
    
    all_client_scores = {client.client_id: [] for client in warmup_clients}
    
    lora_client_map = None
    saved_params = None
    optimal_n_clusters = None
    clustered_clients = None
    aggregated_params = None
    
    for round_idx in tqdm(range(global_rounds), desc="Global Rounds"):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"\n[{current_time}] Starting Global Round {round_idx + 1}/{global_rounds}\n")
        
        print(f"\nGlobal Round {round_idx + 1}/{global_rounds}")
        
        if round_idx < round_warmup:
            print(f"Running warmup phase (round {round_idx+1}/{round_warmup})")
            
            with open(log_file, "a") as f:
                f.write("Starting dummy client warmup\n")
            
            dummy.load_model()
            dummy.local_training(lr=lr, epochs=local_epochs, batch_size=batch_size)
            dummy.unload_model()
            
            client_params = []
            for client in tqdm(warmup_clients, desc="Client Training (Warmup)"):
                with open(log_file, "a") as f:
                    f.write(f"Training Warmup Client {client.client_id} ({client.task_name})...\n")
                
                client.load_model()
                if round_idx > 0 and aggregated_params is not None:
                    client.load_params(aggregated_params[client.client_id])
                
                client.local_training(lr=lr, epochs=local_epochs, batch_size=batch_size)
                params = client.get_lora_params_and_save_by_module(round_id=round_idx, personal_dir=personal_dir)
                client_params.append(params['params'])
                
                if (round_idx + 1) == round_warmup:
                    if saved_params is None:
                        saved_params = {}
                    saved_params[client.client_id] = params['params']
                
                client.unload_model()
            
            with open(log_file, "a") as f:
                f.write("Starting Server Aggregation (Warmup)...\n")
            
            agg_lora_client_map = None
            if (round_idx + 1) == round_warmup:
                with open(log_file, "a") as f:
                    f.write("Computing LoRA client mapping for clustering\n")
                
                lora_client_map, optimal_n_clusters = compute_lora_client_map(
                    warmup_clients, 
                    round_idx, 
                    personal_dir,
                    max_clusters=max_clusters
                )
                agg_lora_client_map = lora_client_map
                
                with open(log_file, "a") as f:
                    f.write(f"LoRA client mapping computed: {lora_client_map}\n")
                    f.write(f"Optimal number of clusters: {optimal_n_clusters}\n")
                
                if task_info is not None and client_datasets is not None:
                    clustered_clients = []
                    for client_id in range(len(warmup_clients)):
                        client_task = task_info[client_id]["task_name"]
                        num_labels = task_info[client_id]["num_labels"]
                        
                        client_cluster = None
                        for cluster_id, cluster_clients in lora_client_map.items():
                            if client_id in cluster_clients:
                                client_cluster = int(cluster_id)
                                break
                        
                        if client_cluster is None:
                            print(f"Warning: Client {client_id} not found in any cluster. Assigning to cluster 0.")
                            client_cluster = 0
                        
                        client = Client(
                            client_id=client_id,
                            task_name=client_task,
                            tokenizer=warmup_clients[client_id].tokenizer,
                            model_name=warmup_clients[client_id].model_name,
                            num_clients=len(warmup_clients),
                            rank=4,
                            lora_n=optimal_n_clusters,
                            adaptive=True,
                            cache_path=output_dir,
                            idx=client_cluster
                        )
                        
                        client.set_dataset(client_datasets[client_id], num_labels)
                        
                        clustered_clients.append(client)
                    
                    clients = clustered_clients
                    
                    server = Server(clients_num=len(clients))
                    
                    with open(log_file, "a") as f:
                        f.write(f"Initialized {len(clients)} clustered clients with {optimal_n_clusters} LoRA modules\n")
            
            aggregated_params = server.aggregation_warmup(
                route_aggregation=True,
                params=client_params,
                lora_client_map=agg_lora_client_map
            )
            
            if (round_idx + 1) % 1 == 0:
                with open(log_file, "a") as f:
                    f.write(f"Performing warmup evaluation at round {round_idx + 1}\n")
                
                print(f"\nWarmup Round {round_idx + 1} Evaluation Scores:")
                round_scores = {}
                
                for client in warmup_clients:
                    client_id = client.client_id
                    client.load_model()
                    client.load_params(aggregated_params[client_id])
                    
                    metrics = client.evaluate_model()
                    all_client_scores[client_id].append(metrics)
                    round_scores[client_id] = metrics
                    
                    client.unload_model()
                
                summary_file = os.path.join(personal_dir, f"round_summary_{round_idx+1}.json")
                with open(summary_file, 'w') as f:
                    json.dump(round_scores, f, indent=2)
        
        else:
            print(f"Running clustered training phase (round {round_idx+1-round_warmup}/{global_rounds-round_warmup})")
            
            if clients is None:
                with open(log_file, "a") as f:
                    f.write("ERROR: Clustered clients not initialized. This should not happen.\n")
                raise RuntimeError("Clustered clients not initialized")
            
            if round_idx == round_warmup:
                with open(log_file, "a") as f:
                    f.write("Transitioning from warmup to clustered training\n")
                    f.write(f"LoRA client mapping: {lora_client_map}\n")
                
                for client in clients:
                    client.load_model()
                    
                    if client.client_id in saved_params:
                        warmed_params = {}
                        client_group = client.idx
                        
                        for name, param in saved_params[client.client_id].items():
                            if 'lora_A0' in name and client_group is not None:
                                new_name = name.replace('lora_A0', f'lora_A{client_group}')
                                warmed_params[new_name] = param
                            elif 'lora_B0' in name and client_group is not None:
                                new_name = name.replace('lora_B0', f'lora_B{client_group}')
                                warmed_params[new_name] = param
                            elif 'lora_route' in name:
                                continue
                            else:
                                warmed_params[name] = param
                        
                        client.local_model.load_state_dict(warmed_params, strict=False)
                    
                    client.unload_model()
            
            client_params = []
            for client in tqdm(clients, desc="Client Training (Clustered)"):
                with open(log_file, "a") as f:
                    f.write(f"Training Clustered Client {client.client_id} ({client.task_name})...\n")
                
                client.load_model()
                if round_idx > round_warmup:
                    client.load_params(aggregated_params[client.client_id])
                
                client.local_training(lr=lr, epochs=local_epochs, batch_size=batch_size, 
                                     lora_client_map=lora_client_map)
                params = client.get_lora_params()
                client_params.append(params['params'])
                
                client.unload_model()
            
            with open(log_file, "a") as f:
                f.write("Starting Server Aggregation (Clustered)...\n")
            
            aggregated_params = server.aggregation(
                route_aggregation=True,
                params=client_params,
                lora_client_map=lora_client_map
            )
            
            if (round_idx + 1) % 1 == 0:
                with open(log_file, "a") as f:
                    f.write(f"Performing clustered evaluation at round {round_idx + 1}\n")
                
                print(f"\nClustered Round {round_idx + 1} Evaluation Scores:")
                round_scores = {}
                
                for client in clients:
                    client_id = client.client_id
                    client.load_model()
                    client.load_params(aggregated_params[client_id])
                    
                    metrics = client.evaluate_model()
                    all_client_scores[client_id].append(metrics)
                    round_scores[client_id] = metrics
                    
                    client.unload_model()
                
                summary_file = os.path.join(personal_dir, f"round_summary_{round_idx+1}.json")
                with open(summary_file, 'w') as f:
                    json.dump(round_scores, f, indent=2)
        
        with open(log_file, "a") as f:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{current_time}] Completed Global Round {round_idx + 1}/{global_rounds}\n")
            f.write("-" * 50 + "\n")
    
    with open(os.path.join(personal_dir, "training_history.json"), 'w') as f:
        json.dump({
            "client_scores": all_client_scores,
            "optimal_n_clusters": optimal_n_clusters,
            "lora_client_map": {str(k): v for k, v in lora_client_map.items()} if lora_client_map else None
        }, f, indent=2)
    
    return all_client_scores


def parse_args():
    parser = argparse.ArgumentParser(description="FedLEASE: Adaptive LoRA Experts Allocation and Selection")
    
    parser.add_argument("--model_name", type=str, default="roberta-large",
                        help="Pre-trained model name")
    parser.add_argument("--tasks", nargs="+", default=["sst2", "sst2", "sst2", "sst2", 
                                                         "qnli", "qnli", "qnli", "qnli",
                                                         "mrpc", "mrpc", "mrpc", "mrpc",
                                                         "qqp", "qqp", "qqp", "qqp"],
                        help="List of tasks for each client")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--global_rounds", type=int, default=25,
                        help="Number of global federated rounds")
    parser.add_argument("--local_epochs", type=int, default=2,
                        help="Number of local training epochs")
    parser.add_argument("--warmup_rounds", type=int, default=5,
                        help="Number of warmup rounds before clustering")
    parser.add_argument("--lr", type=float, default=3e-3,
                        help="Learning rate")
    parser.add_argument("--rank", type=int, default=4,
                        help="LoRA rank")
    parser.add_argument("--max_clusters", type=int, default=4,
                        help="Maximum number of LoRA expert clusters")
    parser.add_argument("--train_samples", type=int, default=1000,
                        help="Training samples per client")
    parser.add_argument("--test_samples", type=int, default=200,
                        help="Test samples per client")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    task_name_list = args.tasks
    client_num = len(task_name_list)
    
    output_dir = os.path.join(args.output_dir, f"{args.model_name.replace('/', '_')}_multi_task_federated_{client_num}")
    
    print(f"Running federated learning with multi-task datasets: {task_name_list}")
    print(f"Number of clients: {client_num}")
    print(f"Model: {args.model_name}")
    print(f"Output directory: {output_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    client_datasets, task_info = partition_multi_task_dataset(
        task_name_list=task_name_list,
        tokenizer=tokenizer,
        alpha=100000,
        train_samples_per_client=args.train_samples,
        test_samples_per_client=args.test_samples,
        seed=args.seed
    )
    
    dummy_task = task_name_list[0]
    dummy_num_labels = task_info[0]["num_labels"]
    
    dummy = WarmupClient(
        client_id=client_num,
        task_name=dummy_task,
        tokenizer=tokenizer,
        model_name=args.model_name,
        num_clients=client_num,
        rank=args.rank,
        cache_path=output_dir
    )
    dummy.set_dataset(client_datasets[0], dummy_num_labels)
    
    warmup_clients = []
    for client_id in range(client_num):
        client_task = task_info[client_id]["task_name"]
        num_labels = task_info[client_id]["num_labels"]
        
        client = WarmupClient(
            client_id=client_id,
            task_name=client_task,
            tokenizer=tokenizer,
            model_name=args.model_name,
            num_clients=client_num,
            rank=args.rank,
            cache_path=output_dir
        )
        client.set_dataset(client_datasets[client_id], num_labels)
        warmup_clients.append(client)
    
    warmup_server = Server(clients_num=len(warmup_clients))
    
    train_result = train_federated(
        dummy=dummy,
        clients=warmup_clients,
        server=warmup_server,
        global_rounds=args.global_rounds,
        local_epochs=args.local_epochs,
        output_dir=output_dir,
        lr=args.lr,
        round_warmup=args.warmup_rounds,
        max_clusters=args.max_clusters,
        task_info=task_info,
        client_datasets=client_datasets,
        batch_size=args.batch_size
    )
    
    print("\nTraining completed!")
    print("Final Evaluation Scores for each client:", train_result)


if __name__ == "__main__":
    main()
