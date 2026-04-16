import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmForMaskedLM
from peft import LoraConfig, get_peft_model
import json

class DualEngineAntigenLM(nn.Module):
    def __init__(self, config_file):
        super().__init__()
        
        # Load custom configuration
        import json
        with open(config_file, 'r') as f:
            self.custom_config = json.load(f)
            
        base_name = self.custom_config.get('base_model_name', "facebook/esm2_t33_650M_UR50D")
        
        print(f"Loading Base Teacher model: {base_name}")
        # 1. Teacher model (Frozen Thermodynamics Anchor)
        self.teacher = EsmForMaskedLM.from_pretrained(base_name)
        self.teacher.eval() # Keep dropout disabled
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        print(f"Loading Base Student model: {base_name}")
        # 2. Student Sub-Network (Epitope Escape Learner)
        student_base = EsmForMaskedLM.from_pretrained(base_name)
        
        # Inject LoRA Adapters fully into targeted layers
        lora_config = LoraConfig(
            r=self.custom_config.get('lora_r', 8),
            lora_alpha=self.custom_config.get('lora_alpha', 32),
            lora_dropout=self.custom_config.get('lora_dropout', 0.05),
            target_modules=self.custom_config.get('target_modules', ["query", "key", "value", "out_proj"]),
            modules_to_save=self.custom_config.get('modules_to_save', ["lm_head"]),
            bias="none"
        )
        
        self.student = get_peft_model(student_base, lora_config)
        self.student.print_trainable_parameters()
        
        # Expose Native HuggingFace config to Trainer to avoid AttributeErrors (use_cache, etc.)
        self.config = getattr(self.student, "config", getattr(student_base, "config", None))
        
        # Distillation loss hyperparameters
        self.kd_temperature = self.custom_config.get('kd_temperature', 2.0)
        self.lambda_mlm = self.custom_config.get('lambda_mlm', 1.0)
        self.lambda_kd = self.custom_config.get('lambda_kd', 1.0)

    def gradient_checkpointing_enable(self, **kwargs):
        """
        Forward gradient checkpointing enablement to the underlying wrapped model.
        """
        if hasattr(self.student, "gradient_checkpointing_enable"):
            self.student.gradient_checkpointing_enable(**kwargs)

    def train(self, mode=True):
        """
        Override train mode to ensure Teacher always stays in eval() mode.
        """
        super().train(mode)
        self.teacher.eval()

    def forward(self, input_ids, attention_mask, labels, structure_mask=None, **kwargs):
        """
        Forward logic enforcing spatial decoupling:
        - MLM Loss on Epiope-biased masked tokens (from Student labels tensor)
        - KD Loss on Unmasked structure backbone (KL Div between Teacher & Student)
        """
        # Student Forward
        student_outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels  # Only dynamically masked positions have valid indices; rest are -100
        )
        
        student_logits = student_outputs.logits
        mlm_loss = student_outputs.loss # Internal HuggingFace CrossEntropy exclusively on dynamic mask labels
        
        loss = self.lambda_mlm * mlm_loss
        kd_loss = torch.tensor(0.0, device=student_logits.device)
        
        # Online Distillation Flow
        if structure_mask is not None and self.lambda_kd > 0.0:
            with torch.no_grad():
                teacher_outputs = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits.detach()
                
            # Filter solely for unmasked structural backbone tokens
            active_structure_idx = structure_mask.view(-1)
            
            s_logits_flat = student_logits.view(-1, student_logits.size(-1))[active_structure_idx]
            t_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))[active_structure_idx]
            
            if s_logits_flat.size(0) > 0:
                # Softmax with temperature scaling for softer target distributions
                student_log_probs = F.log_softmax(s_logits_flat / self.kd_temperature, dim=-1)
                teacher_probs = F.softmax(t_logits_flat / self.kd_temperature, dim=-1)
                
                kd_temp_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
                
                # Multiply by T^2 as per standard Distillation algorithms
                kd_loss = kd_temp_loss * (self.kd_temperature ** 2)
                
                # Fuse Engines
                loss = loss + self.lambda_kd * kd_loss
                
        return {
            "loss": loss,
            "mlm_loss": mlm_loss,
            "kd_loss": kd_loss,
            "student_logits": student_logits
        }
