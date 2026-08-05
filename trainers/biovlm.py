import os.path as osp
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as T

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from models import clip
from models.open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

def load_clip_to_cpu(cfg):
    print("\n\nUsing BioMedCLIP ...\n\n")
    model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model.eval()                     
    return model

import json
def load_json(filename):
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'r') as fp:
        return json.load(fp)
    

def wordify(string):
    word = string.replace('_', ' ')
    return word
 
def replace_underscores(classnames):
    return [classname.replace('_', ' ') for classname in classnames]
 
def replace_commas(descriptions):
    return [description.replace(',', ' ') for description in descriptions]
 
def make_descriptor_sentence(descriptor):
    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f"which is {descriptor}"
    elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
        return f"which {descriptor}"
    elif descriptor.startswith('used'):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"
    
    
def modify_descriptor(descriptor, apply_changes=True):
    if apply_changes:
        return make_descriptor_sentence(descriptor)
    return descriptor


def load_gpt_descriptions(cfg, json_file, classnames):
    gpt_descriptions = load_json(json_file)
    unmodify_dict = {}
    
    filtered_descriptions = {k: v for k, v in gpt_descriptions.items() if k in replace_underscores(classnames)}
    
    for i, (k, v) in enumerate(filtered_descriptions.items()):
        if len(v) == 0:
            v = ['']
            
        word_to_add = wordify(k)
        if cfg.DATASET.NAME == 'PathMNIST':
            prefix = "a colon pathology of"
            suffix = "type of colorectal cancer"
        if cfg.DATASET.NAME == 'DermaMNIST':
            prefix = "a dermatoscopic image of"
            suffix = "type of skin lesion"
        if cfg.DATASET.NAME == 'OCTMNIST':
            prefix = "an optical coherence tomography image of"
            suffix = "retina"
        if cfg.DATASET.NAME == 'PneumoniaMNIST':
            prefix = "a pediatric X-Ray image of"
            suffix = "chest"
        if cfg.DATASET.NAME == 'RetinaMNIST':
            prefix = "a fundus image of"
            suffix = ""
        if cfg.DATASET.NAME == 'BreastMNIST':
            prefix = "an ultrasound image of"
            suffix = "tumor found in breast tissue"
        if cfg.DATASET.NAME == 'BloodMNIST':
            prefix = "a microscopic image of"
            suffix = "type of blood cell"
        if cfg.DATASET.NAME == 'TissueMNIST':
            prefix = "a microscopic image of"
            suffix = "type of kidney cortex cell"
        if cfg.DATASET.NAME == 'OrganAMNIST':
            prefix = "a cropped 2D image from the center slices of the bounding boxes of 3D computed tomography image of"
            suffix = "organ in axial views"
        if cfg.DATASET.NAME == 'OrganCMNIST':
            prefix = "a cropped 2D image from the center slices of the bounding boxes of 3D computed tomography image of"
            suffix = "organ in coronal views"
        if cfg.DATASET.NAME == 'OrganSMNIST':
            prefix = "a cropped 2D image from the center slices of the bounding boxes of 3D computed tomography image of"
            suffix = "organ in sagittal views"
            
        if cfg.DATASET.NAME == 'BTMRI':
            prefix = "a magnetic resonance imaging of"
            suffix = "of brain"
        if cfg.DATASET.NAME == 'BUSI':
            prefix = "aN ultrasound image of"
            suffix = "of breast"
        if cfg.DATASET.NAME == 'CHMNIST':
            prefix = "a colorectal histopathology image of"
            suffix = ""
        if cfg.DATASET.NAME == 'COVID_19':
            prefix = "a chest X-Ray image of"
            suffix = ""
        if cfg.DATASET.NAME == 'CTKidney':
            prefix = "a computerized tomography image of"
            suffix = "of kidney"
        if cfg.DATASET.NAME == 'DermaMNISTBIO':
            prefix = "a dermatoscopic image of"
            suffix = "type of skin lesion"
        if cfg.DATASET.NAME == 'KneeXray':
            prefix = "a knee X-Ray image of"
            suffix = ""
        if cfg.DATASET.NAME == 'Kvasir':
            prefix = "an endoscopy image of"
            suffix = "of colon"
        if cfg.DATASET.NAME == 'LungColon':
            prefix = "a histopathology image of"
            suffix = "of lung and colon"
        if cfg.DATASET.NAME == 'OCTMNISTBIO':
            prefix = "an optical coherence tomography image of"
            suffix = "of retina"
        if cfg.DATASET.NAME == 'RETINA':
            prefix = "a fundus photography of"
            suffix = "of retina"
            
            
        build_descriptor_string = lambda item: f"{prefix} {word_to_add} {suffix}, {modify_descriptor(item)}"
    
        unmodify_dict[k] = {build_descriptor_string(item): item for item in v}
        filtered_descriptions[k] = [build_descriptor_string(item) for item in v]
    
    return filtered_descriptions

def aggregate_similarity(similarity_matrix_chunk, aggregation_method='mean'):
    if aggregation_method == 'max': return similarity_matrix_chunk.max(dim=1)[0]
    elif aggregation_method == 'sum': return similarity_matrix_chunk.sum(dim=1)
    elif aggregation_method == 'mean': return similarity_matrix_chunk.mean(dim=1)
    else: raise ValueError("Unknown aggregate_similarity")
    
    
def strong_aug(image):
    """
    Apply strong augmentation to a batch of images.
    Args:
        img_batch (torch.Tensor): Batch of images of shape (32, 3, 224, 224)
    Returns:
        torch.Tensor: Augmented image batch of shape (32, 3, 224, 224)
    """
    # batch_size, channels, height, width = img_batch.shape
    # assert (batch_size, channels, height, width) == (32, 3, 224, 224), "Input shape must be (32, 3, 224, 224)"
    
    transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.5)
    ])
    
    augmented_batch = torch.stack([transform(img) for img in image])
    return augmented_batch


def weak_aug(image):
    """
    Apply weak augmentation to a batch of images.
    Args:
        img_batch (torch.Tensor): Batch of images of shape (32, 3, 224, 224)
    Returns:
        torch.Tensor: Augmented image batch of shape (32, 3, 224, 224)
    """
    # batch_size, channels, height, width = img_batch.shape
    # assert (batch_size, channels, height, width) == (32, 3, 224, 224), "Input shape must be (32, 3, 224, 224)"
    
    transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5)
    ])
    
    augmented_batch = torch.stack([transform(img) for img in image])
    return augmented_batch


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.visual.head.proj.weight.dtype

        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        # ctx_dim = medclip_model.text_model.projection_head.weight.shape[0]
        ctx_dim = 768
        clip_imsize = 224 # BioMedCLIP's default image size

        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(10, n_ctx, ctx_dim, dtype=dtype)

            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")


        # print("\n\nUsing Random Context Initialization\n\n")
        # self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        print("\n\nUsing Pre-trained Context Initialization\n\n")
        self.ctx = nn.Parameter(ctx_vectors)
        # self.ctx = nn.Parameter(torch.load(os.path.join(os.getcwd(), 'models', 'ctx_vectors', f'ctx_{cfg.MODEL_NAME}_{cfg.DATASET_NAME}_s{cfg.SEED}.pt')))
        # Note: This context is pre-trained using the clean images of the few-shot train dataset (i.e. with POISON_PERCENTAGE=0)


        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(tokenizer.tokenizer.encode(name))-2 for name in classnames]   # [CLS] and [SEP] are not counted
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        
        context_length = 256
        prompts_tokens = tokenizer(prompts, context_length=context_length)
        self.prompts_attention_mask = (prompts_tokens != clip_model.text.config.pad_token_id).long()


        with torch.no_grad():
            prompts_tokens_embeddings = clip_model.text.transformer.embeddings(input_ids=prompts_tokens).type(dtype) # [n_cls, 256, 768]
            prompts_tokens_embeddings = prompts_tokens_embeddings.unsqueeze(0).expand(10, -1, -1, -1)


        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", prompts_tokens_embeddings[:, :, :1, :])  # CLS
        self.register_buffer("token_suffix", prompts_tokens_embeddings[:, :, 1 + n_ctx :, :])  # CLASS_NAMES_TOKENS, SEP, PAD

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        

    def forward(self):
        
        ctx = self.ctx
        
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
            

        prefix = self.token_prefix
        suffix = self.token_suffix
     

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=2,
            )
        else:
            raise ValueError


        return prompts, self.prompts_attention_mask


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.text_model = clip_model.text
       
    def forward(self, prompts_embeddings, prompts_attention_mask, normalize=False):
        out = self.text_model.transformer(inputs_embeds=prompts_embeddings, attention_mask=prompts_attention_mask)
        pooled_out = self.text_model.pooler(out, prompts_attention_mask)
        projected =  self.text_model.proj(pooled_out)
        return F.normalize(projected, dim=-1) if normalize else projected

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.clip_model = clip_model
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        #self.text_proj = nn.Linear(768, 512)
        self.dataset = cfg.DATASET.NAME
        self.device = device
        self.dtype = clip_model.text.transformer.embeddings.word_embeddings.weight.dtype
        self.classnames = classnames
        self.context_length = 256
        self.dtype = clip_model.text.transformer.embeddings.word_embeddings.weight.dtype
        if cfg.DATASET.NAME == 'PathMNIST':
            self.gpt_descriptions = load_gpt_descriptions(cfg, './attributes_medmnist/pathmnist.json',self.classnames)
        if cfg.DATASET.NAME == 'DermaMNIST':
            self.gpt_descriptions = load_gpt_descriptions(cfg, './attributes_medmnist/dermamnist.json',self.classnames)
        if cfg.DATASET.NAME == 'OCTMNIST':
            self.gpt_descriptions = load_gpt_descriptions(cfg, './attributes_medmnist/octmnist.json',self.classnames)
        if cfg.DATASET.NAME == 'PneumoniaMNIST':
            self.gpt_descriptions = load_gpt_descriptions(cfg, './attributes_medmnist/pneumoniamnist.json',self.classnames)
        if cfg.DATASET.NAME == 'RetinaMNIST':
            self.gpt_descriptions = load_gpt_descriptions(cfg, './attributes_medmnist/retinamnist.json',self.classnames)
        if cfg.DATASET.NAME == 'BreastMNIST':
            self.gpt_descriptions = load_gpt_descriptions(cfg, './attributes_medmnist/breastmnist.json',self.classnames)
        if cfg.DATASET.NAME == 'BloodMNIST':
            self.gpt_descriptions = load_gpt_descriptions(cfg, './attributes_medmnist/bloodmnist.json',self.classnames)
        if cfg.DATASET.NAME == 'TissueMNIST':
            self.gpt_descriptions = load_gpt_descriptions(cfg, './attributes_medmnist/tissuemnist.json',self.classnames)
        if cfg.DATASET.NAME == 'OrganAMNIST':
            self.gpt_descriptions = load_gpt_descriptions(cfg, './attributes_medmnist/organamnist.json',self.classnames)
        if cfg.DATASET.NAME == 'OrganCMNIST':
            self.gpt_descriptions = load_gpt_descriptions(cfg, './attributes_medmnist/organcmnist.json',self.classnames)
        if cfg.DATASET.NAME == 'OrganSMNIST':
            self.gpt_descriptions = load_gpt_descriptions(cfg, './attributes_medmnist/organsmnist.json',self.classnames)
        
        if cfg.DATASET.NAME == 'BTMRI':
            self.gpt_descriptions = load_gpt_descriptions(cfg, './attributes_bio/btmri.json',self.classnames)
        if cfg.DATASET.NAME == 'BUSI':
            self.gpt_descriptions = load_gpt_descriptions(cfg, './attributes_bio/busi.json',self.classnames)
        if cfg.DATASET.NAME == 'CHMNIST':
            self.gpt_descriptions = load_gpt_descriptions(cfg, './attributes_bio/chmnist.json',self.classnames)
        if cfg.DATASET.NAME == 'COVID_19':
            self.gpt_descriptions = load_gpt_descriptions(cfg, './attributes_bio/covid19.json',self.classnames)
        if cfg.DATASET.NAME == 'CTKidney':
            self.gpt_descriptions = load_gpt_descriptions(cfg, './attributes_bio/ctkidney.json',self.classnames)
        if cfg.DATASET.NAME == 'DermaMNISTBIO':
            self.gpt_descriptions = load_gpt_descriptions(cfg, './attributes_bio/dermamnist.json',self.classnames)
        if cfg.DATASET.NAME == 'KneeXray':
            self.gpt_descriptions = load_gpt_descriptions(cfg, './attributes_bio/kneexray.json',self.classnames)
        if cfg.DATASET.NAME == 'Kvasir':
            self.gpt_descriptions = load_gpt_descriptions(cfg, './attributes_bio/kvasir.json',self.classnames)
        if cfg.DATASET.NAME == 'LungColon':
            self.gpt_descriptions = load_gpt_descriptions(cfg, './attributes_bio/lungcolon.json',self.classnames)
        if cfg.DATASET.NAME == 'OCTMNISTBIO':
            self.gpt_descriptions = load_gpt_descriptions(cfg, './attributes_bio/octmnist.json',self.classnames)
        if cfg.DATASET.NAME == 'RETINA':
            self.gpt_descriptions = load_gpt_descriptions(cfg, './attributes_bio/retina.json',self.classnames)

    def forward(self, image):
        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
        
        
        prompts_embeddings, prompts_attention_mask = self.prompt_learner()
        
        text_featureslist = []
        for i in range(prompts_embeddings.shape[0]):  
            prompts_embeddings_i = prompts_embeddings[i] 
            textfeatures = self.text_encoder(prompts_embeddings_i, prompts_attention_mask.cuda(), normalize=True)
            text_featureslist.append(textfeatures)
        text_features = torch.stack(text_featureslist, dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logits_list = []
        logit_scale = self.logit_scale.exp()
        for i in range(text_features.shape[0]):  
            text_features_i = text_features[i] 
            logits_i = logit_scale * image_features @ text_features_i.t()
            logits_list.append(logits_i)
        logits = torch.stack(logits_list, dim=0)
        logits = logits.permute(1,0,2)
        batch_size = logits.shape[0]
        
        if self.prompt_learner.training:
            
            strong_image_features = self.image_encoder(strong_aug(image))
            strong_image_features = strong_image_features / strong_image_features.norm(dim=-1, keepdim=True)
        
            weak_image_features = self.image_encoder(weak_aug(image))
            weak_image_features = weak_image_features / weak_image_features.norm(dim=-1, keepdim=True)
            
        
            gpt_featurelist = []
            for i, (k, v) in enumerate(self.gpt_descriptions.items()): 
                texts = tokenizer(v, context_length=self.context_length).cuda()
                gptfeatures = self.clip_model.text(texts)
                gpt_featurelist.append(gptfeatures)

            gpt_features = torch.stack(gpt_featurelist)
            gpt_features = gpt_features / gpt_features.norm(dim=-1, keepdim=True)
            gpt_features = gpt_features.permute(1, 0, 2)
            
        
            gptlogits_list = []
            gptauglogits_list = []
            for i in range(gpt_features.shape[0]):  
                gpt_features_i = gpt_features[i] 
                gptlogits_i = logit_scale * image_features @ gpt_features_i.t()
                strong_gptlogits_i = logit_scale * strong_image_features @ gpt_features_i.t()
                weak_gptlogits_i = logit_scale * weak_image_features @ gpt_features_i.t()
                gptlogits_list.append(gptlogits_i)
                gptauglogits_list.append(strong_gptlogits_i)
                gptauglogits_list.append(weak_gptlogits_i)
            gptlogits = torch.stack(gptlogits_list, dim=0)
            gptauglogits = torch.stack(gptauglogits_list, dim=0)
            
            gptlogits = gptlogits.permute(1,0,2)
            gptauglogits = gptauglogits.permute(1,0,2)
            
            
            selected_output_list = []
            entropy_loss_list = []
            selected_gptoutput_list = []
            gptentropy_loss_list = []
            selected_gptaugoutput_list = []
            gptaugentropy_loss_list = []
            for i in range(batch_size):
                logits_i = logits[i]           
                output_mean = output.mean(dim=0)
                selected_output_list.append(output_mean)
                ent_loss = avg_entropy(output)
                entropy_loss_list.append(ent_loss)
            
                gptlogits_i = gptlogits[i]
                gptoutput, gptselected_idx = select_confident_samples(gptlogits_i, top=0.5) 
                gptoutput_mean = gptoutput.mean(dim=0)
                selected_gptoutput_list.append(gptoutput_mean)
                gptent_loss = avg_entropy(gptoutput)
                gptentropy_loss_list.append(gptent_loss)
                
                
                gptauglogits_i = gptauglogits[i]
                gptaugoutput, gptselected_idx = select_confident_samples(gptauglogits_i, top=0.5) 
                gptaugoutput_mean = gptaugoutput.mean(dim=0)
                selected_gptaugoutput_list.append(gptaugoutput_mean)
                gptaugent_loss = avg_entropy(gptaugoutput)
                gptaugentropy_loss_list.append(gptaugent_loss)
            
            final_logits = torch.stack(selected_output_list, dim=0)  # Shape: [32, 512]
            loss_ent = torch.mean(torch.tensor(entropy_loss_list))  # Scalar value
        
            final_gptlogits = torch.stack(selected_gptoutput_list, dim=0)  # Shape: [32, 512]
            loss_gptent = torch.mean(torch.tensor(gptentropy_loss_list))
            
            final_gptauglogits = torch.stack(selected_gptaugoutput_list, dim=0)  # Shape: [32, 512]
            loss_gptaugent = torch.mean(torch.tensor(gptaugentropy_loss_list))
           
        
            loss_kdsp = F.kl_div(
                    F.log_softmax(final_logits, dim=1),
                    F.log_softmax(final_gptlogits, dim=1),
                    reduction='sum',
                    log_target=True
                ) / logits.numel()
            
            loss_kdsp_aug = F.kl_div(
                    F.log_softmax(final_logits, dim=1),
                    F.log_softmax(final_gptauglogits, dim=1),
                    reduction='sum',
                    log_target=True
                ) / logits.numel()
        
        
            cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
            score = cos(text_features,gpt_features)
            loss_cos = 1.0-torch.mean(score)
            
            loss_ent = loss_ent.to(self.device)
            loss_gptent = loss_gptent.to(self.device)
            loss_gptaugent = loss_gptaugent.to(self.device)
            loss_kdsp = loss_kdsp.to(self.device)
            loss_kdsp_aug = loss_kdsp_aug.to(self.device)
            loss_cos = loss_cos.to(self.device)
        
        

            return final_logits, loss_ent, loss_gptent, loss_gptaugent, loss_kdsp, loss_kdsp_aug, loss_cos
        else:
            selected_output_list = []
            for i in range(batch_size):
                logits_i = logits[i]
    
                output, selected_idx = select_confident_samples(logits_i, top=0.5) 
                output_mean = output.mean(dim=0)
                selected_output_list.append(output_mean)
            final_logits = torch.stack(selected_output_list, dim=0)
            
            return final_logits
            

@TRAINER_REGISTRY.register()
class BioVLM(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, device=self.device)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler(enabled=True) if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            #self.model = nn.DataParallel(self.model)
            self.model.text_encoder = nn.DataParallel(self.model.text_encoder)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast(enabled=True):
                output, loss_ent, loss_gptent, loss_gptaugent, loss_kdsp, loss_kdsp_aug, loss_cos = self.model(image)
                loss = F.cross_entropy(output, label) + loss_ent + loss_gptent + loss_gptaugent + loss_kdsp + loss_kdsp_aug + loss_cos
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, loss_ent, loss_gptent, loss_gptaugent, loss_kdsp, loss_kdsp_aug, loss_cos = self.model(image)
            loss_ce = F.cross_entropy(output, label)
            loss = loss_ce + 0.5*loss_ent + 0.5*loss_gptent + 0.5*loss_gptaugent + loss_kdsp + loss_kdsp_aug + 0.5*loss_cos
            self.model_backward_and_update(loss)
            
        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
    
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


