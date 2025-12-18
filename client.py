import os
import torch
import json
import numpy as np
import gc
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from peft import LoraConfig, TaskType, get_peft_model


class Client:
    def __init__(self, client_id, task_name, tokenizer, model_name, num_clients, rank=8, lora_n=4, adaptive=False, cache_path="./output", idx=None):
        self.client_id = client_id
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.num_clients = num_clients
        self.rank = rank
        self.lora_n = lora_n
        self.adaptive = adaptive
        self.cache_path = cache_path
        self.local_model = None
        self.current_params = None
        self.datasets = None
        self.num_labels = None
        self.idx = idx
        
    def set_dataset(self, dataset, num_labels):
        self.datasets = dataset
        self.num_labels = num_labels
    
    def load_model(self):
        if self.local_model is None:
            self.local_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_path,
                num_labels=self.num_labels,
            )
            
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                target_modules=["query", "value"],
                inference_mode=False,
                r=self.rank,
                lora_alpha=16,
                lora_dropout=0.05,
                lora_nums=self.lora_n,
                adaptive=self.adaptive,
                idx=self.idx,
                k=self.lora_n,
                bias="none"
            )
            
            self.local_model = get_peft_model(self.local_model, peft_config)
            
            if self.current_params is not None:
                self.load_params(self.current_params)
    
    def unload_model(self):
        if self.local_model is not None:
            self.current_params = self.get_lora_params()['params']
            del self.local_model
            self.local_model = None
            torch.cuda.empty_cache()
            gc.collect()
        
    def get_lora_params(self):
        lora_params = {
            'client_id': self.client_id,
            'params': {}
        }
        for name, param in self.local_model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name or 'lora_route' in name or 'classifier' in name:
                lora_params['params'][name] = param.data.clone()
        return lora_params
    
    def get_lora_params_and_save_by_module(self, round_id, personal_dir):
        lora_params = {
            'client_id': self.client_id,
            'params': {}
        }

        target_modules = ["query", "value"]
        lora_A_dict = {module: [] for module in target_modules}
        lora_B_dict = {module: [] for module in target_modules}

        for name, param in self.local_model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name or 'lora_route' in name or 'classifier' in name:
                lora_params['params'][name] = param.data.clone()

            for module in target_modules:
                if module in name:
                    if 'lora_A0' in name:
                        lora_A_dict[module].append(param.data.clone().cpu().numpy())
                    elif 'lora_B0' in name:
                        lora_B_dict[module].append(param.data.clone().cpu().numpy())

        param_dir = os.path.join(personal_dir, "lora_params")
        os.makedirs(param_dir, exist_ok=True)

        for module in target_modules:
            if lora_A_dict[module]:  
                np.save(os.path.join(param_dir, f'{module}_lora_A_client_{self.client_id}_{round_id}.npy'), np.array(lora_A_dict[module]))
            if lora_B_dict[module]:  
                np.save(os.path.join(param_dir, f'{module}_lora_B_client_{self.client_id}_{round_id}.npy'), np.array(lora_B_dict[module]))

        return lora_params
    
    def load_params(self, params_or_path):
        if isinstance(params_or_path, dict):
            params_to_load = params_or_path['params'] if 'params' in params_or_path else params_or_path
            self.local_model.load_state_dict(params_to_load, strict=False)
        else:
            self.local_model.load_adapter(params_or_path, adapter_name="default")
            

    def local_training(self, lr=2e-4, epochs=1, batch_size=32, gradient_accumulation_steps=1, lora_client_map=None):
        self.local_model.train()

        if lora_client_map is None:
            raise ValueError("lora_client_map is required for local_training after warmup")

        client_lora_group = None
        for lora_idx, client_indices in lora_client_map.items():
            if self.client_id in client_indices:
                client_lora_group = int(lora_idx)
                break

        if client_lora_group is None:
            print(f"Client {self.client_id} is a dummy client or not found in lora_client_map")
            print(f"Training all LoRA modules for client {self.client_id}")

            for name, param in self.local_model.named_parameters():
                if 'lora_A' in name or 'lora_B' in name or 'lora_route' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            print(f"Client {self.client_id} belongs to LoRA group {client_lora_group}")
            for name, param in self.local_model.named_parameters():
                lora_a_pattern = f'lora_A{client_lora_group}'
                lora_b_pattern = f'lora_B{client_lora_group}'

                if lora_a_pattern in name or lora_b_pattern in name or 'lora_route' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        trainable_params = [p for p in self.local_model.parameters() if p.requires_grad]
        print(f"Number of trainable parameters: {len(trainable_params)}")
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found!")

        training_args = TrainingArguments(
            output_dir=f"{self.cache_path}/{self.rank}_{self.lora_n}_proposed/client_{self.client_id}_checkpoints",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=True,
            logging_steps=5,
            optim="adamw_torch",
            weight_decay=0.05,
            evaluation_strategy="no",
            save_strategy="no",
            save_total_limit=1,
            remove_unused_columns=False,
            gradient_checkpointing=False
        )

        trainer = Trainer(
            model=self.local_model,
            args=training_args,
            train_dataset=self.datasets["train"],
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer)
        )

        trainer.train()
        
    def evaluate_model(self, output_file=None):
        self.local_model.eval()
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return {"accuracy": (predictions == labels).astype(np.float32).mean().item()}

        eval_args = TrainingArguments(
            output_dir=f"{self.cache_path}/temp_eval_output",
            per_device_eval_batch_size=256,
            fp16=True,  
            report_to="none" 
        )

        metrics = Trainer(
            model=self.local_model,
            args=eval_args,
            eval_dataset=self.datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=compute_metrics 
        ).evaluate()

        print(f"Evaluation metrics for client {self.client_id} on validation dataset:")
        print(metrics)

        if output_file:
            with open(output_file, 'w') as f:
                json.dump({
                    'client_id': self.client_id,
                    'task': self.task_name,
                    'dataset_type': 'validation',
                    'metrics': metrics
                }, f, indent=2)
            print(f'The output file is stored at {output_file}')

        return metrics



class WarmupClient(Client):
    def __init__(self, client_id, task_name, tokenizer, model_name, num_clients, rank=8, cache_path="./output"):
        
        super().__init__(client_id, task_name, tokenizer, model_name, num_clients, rank, lora_n=1, adaptive=False, cache_path=cache_path)
        
    def load_model(self):
        
        if self.local_model is None:
            self.local_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_path,
                num_labels=self.num_labels,
            )
            
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                target_modules=["query", "value"],
                inference_mode=False,
                r=self.rank,
                lora_alpha=16,
                lora_dropout=0.05,
                lora_nums=1,
                adaptive=False, 
                idx=0,  
                k=1, 
                bias="none"
            )
            
            self.local_model = get_peft_model(self.local_model, peft_config)
            
            if self.current_params is not None:
                self.load_params(self.current_params)

    def local_training(self, lr=2e-4, epochs=1, batch_size=32, gradient_accumulation_steps=1):
        self.local_model.train()

        for name, param in self.local_model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        trainable_params = [p for p in self.local_model.parameters() if p.requires_grad]
        print(f"Number of trainable parameters: {len(trainable_params)}")
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found!")

        training_args = TrainingArguments(
            output_dir=f"{self.cache_path}/warmup/client_{self.client_id}_checkpoints",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=True,
            logging_steps=5,
            optim="adamw_torch",
            weight_decay=0.05,
            evaluation_strategy="no",
            save_strategy="no",
            save_total_limit=1,
            remove_unused_columns=False,
            gradient_checkpointing=False
        )

        trainer = Trainer(
            model=self.local_model,
            args=training_args,
            train_dataset=self.datasets["train"],
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer)
        )

        trainer.train()

    def get_lora_params(self):
        lora_params = {
            'client_id': self.client_id,
            'params': {}
        }
        for name, param in self.local_model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name or 'classifier' in name:
                lora_params['params'][name] = param.data.clone()
        return lora_params
    
    def get_lora_params_and_save_by_module(self, round_id, personal_dir):
        lora_params = {
            'client_id': self.client_id,
            'params': {}
        }

        target_modules = ["query", "value"]
        lora_A_dict = {module: [] for module in target_modules}
        lora_B_dict = {module: [] for module in target_modules}

        for name, param in self.local_model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name or 'lora_route' in name or 'classifier' in name:
                lora_params['params'][name] = param.data.clone()

            for module in target_modules:
                if module in name:
                    if 'lora_A0' in name:
                        lora_A_dict[module].append(param.data.clone().cpu().numpy())
                    elif 'lora_B0' in name:
                        lora_B_dict[module].append(param.data.clone().cpu().numpy())

        param_dir = os.path.join(personal_dir, "lora_params")
        print(param_dir)
        os.makedirs(param_dir, exist_ok=True)

        for module in target_modules:
            if lora_A_dict[module]:  
                np.save(os.path.join(param_dir, f'{module}_lora_A_client_{self.client_id}_{round_id}.npy'), np.array(lora_A_dict[module]))
            if lora_B_dict[module]: 
                np.save(os.path.join(param_dir, f'{module}_lora_B_client_{self.client_id}_{round_id}.npy'), np.array(lora_B_dict[module]))

        return lora_params


