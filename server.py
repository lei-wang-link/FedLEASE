import torch
from typing import List, Dict


class Server:
    def __init__(self, clients_num: int, device: str = "cuda"):
        self.clients_num = clients_num
        self.device = device
        self.lora_client_map = None  

    def aggregation_warmup(self, route_aggregation: bool, params: List, lora_client_map=None) -> List[Dict]:
        gpu_params = [
            {k: v.to(self.device) for k, v in client_params.items()}
            for client_params in params
        ]

        num_clients = len(gpu_params)
        aggregated_results = [{} for _ in range(num_clients)]

        final_warmup_round = lora_client_map is not None

        if final_warmup_round:
            self.lora_client_map = lora_client_map
            print("Final warmup round, preparing transition to clustered LoRA")

            for client_idx in range(num_clients):
                for param_name, param_value in gpu_params[client_idx].items():
                    aggregated_results[client_idx][param_name] = param_value

            client_to_group = {}
            for group_idx, clients in lora_client_map.items():
                for client in clients:
                    client_to_group[client] = int(group_idx)

            for group_idx, group_clients in lora_client_map.items():
                group_idx = int(group_idx)

                if not group_clients:
                    continue

                print(f"Processing group {group_idx} with clients {group_clients}")

                valid_clients = [c for c in group_clients if c < num_clients]

                if not valid_clients:
                    continue

                for base_param_name in list(gpu_params[0].keys()):
                    if 'lora_A0' in base_param_name or 'lora_B0' in base_param_name:
                        target_param_name = base_param_name.replace('0', str(group_idx))

                        try:
                            stacked_params = torch.stack([
                                gpu_params[i][base_param_name]
                                for i in valid_clients if base_param_name in gpu_params[i]
                            ]).to(self.device)

                            if stacked_params.size(0) > 0:
                                avg_param = stacked_params.mean(dim=0)

                                for client_idx in group_clients:
                                    if client_idx < num_clients:
                                        aggregated_results[client_idx][target_param_name] = avg_param
                        except Exception as e:
                            print(f"Error aggregating {base_param_name} for group {group_idx}: {e}")
        else:
            for client_idx in range(num_clients):
                for param_name, param_value in gpu_params[client_idx].items():
                    if 'lora_A' in param_name or 'lora_B' in param_name or 'lora_route' in param_name:
                        aggregated_results[client_idx][param_name] = param_value

        return aggregated_results
    
    def aggregation(self, route_aggregation: bool, params: List, lora_client_map=None) -> List[Dict]:
        if lora_client_map is not None:
            self.lora_client_map = lora_client_map

        if self.lora_client_map is None:
            raise ValueError("lora_client_map must be provided for aggregation after warmup phase")

        client_to_group = {}
        for group_idx, clients in self.lora_client_map.items():
            for client in clients:
                client_to_group[client] = group_idx

        gpu_params = [
            {k: v.to(self.device) for k, v in client_params.items()}
            for client_params in params
        ]
        num_clients = len(gpu_params)
        aggregated_results = [{} for _ in range(num_clients)]
        param_names = gpu_params[0].keys()

        for client_idx in range(num_clients):
            for param_name in param_names:

                if 'lora_route' in param_name:
                    if route_aggregation:
                        client_group = client_to_group.get(client_idx)
                        if client_group is not None:
                            group_indices = self.lora_client_map[client_group]
                            stacked_params = torch.stack([
                                gpu_params[i][param_name]
                                for i in group_indices
                            ]).to(self.device)
                            aggregated_results[client_idx][param_name] = stacked_params.mean(dim=0)
                        else:
                            aggregated_results[client_idx][param_name] = gpu_params[client_idx][param_name]
                    else:
                        aggregated_results[client_idx][param_name] = gpu_params[client_idx][param_name]

                elif 'lora_A' in param_name:
                    lora_idx = int(param_name.split('lora_A')[1][0])

                    group_indices = self.lora_client_map.get(str(lora_idx), [])
                    if not group_indices:
                        group_indices = self.lora_client_map.get(lora_idx, [])

                    if group_indices:
                        stacked_params = torch.stack([
                            gpu_params[i][param_name]
                            for i in group_indices if i < len(gpu_params) and param_name in gpu_params[i]
                        ]).to(self.device)
                        if stacked_params.size(0) > 0:
                            aggregated_results[client_idx][param_name] = stacked_params.mean(dim=0)
                        else:
                            aggregated_results[client_idx][param_name] = gpu_params[client_idx][param_name]
                    else:
                        aggregated_results[client_idx][param_name] = gpu_params[client_idx][param_name]

                elif 'lora_B' in param_name:
                    lora_idx = int(param_name.split('lora_B')[1][0])

                    group_indices = self.lora_client_map.get(str(lora_idx), [])
                    if not group_indices:
                        group_indices = self.lora_client_map.get(lora_idx, [])

                    if group_indices:
                        stacked_params = torch.stack([
                            gpu_params[i][param_name]
                            for i in group_indices if i < len(gpu_params) and param_name in gpu_params[i]
                        ]).to(self.device)
                        if stacked_params.size(0) > 0:
                            aggregated_results[client_idx][param_name] = stacked_params.mean(dim=0)
                        else:
                            aggregated_results[client_idx][param_name] = gpu_params[client_idx][param_name]
                    else:
                        aggregated_results[client_idx][param_name] = gpu_params[client_idx][param_name]
                else:
                    aggregated_results[client_idx][param_name] = gpu_params[client_idx][param_name]

        return aggregated_results


