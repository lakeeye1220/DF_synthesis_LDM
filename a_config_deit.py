from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunConfig:
    # WandB setup
    init_project_name: str = "o2m_refactoring"
    init_entity_name: str = "gustn9609"
    
    # Id of the experiment
    exp_description: str = "resnet34_all_update_up_lr_denormalize_4_at_home"
    init_latent_img_file: str = "resnet34_all_update_up_lr_denormalize_4_at_home"
    custom_root: str = "after_refactoring"
    
    # Exp setup
    class_index: int = 283
    train: bool = True
    evaluate: bool = True

    # Whether to use Stable Diffusion v2.1
    sd_2_1: bool = True

    # the classifier (Options: inet (ImageNet), inat (iNaturalist), cub (CUB200))
    classifier: str = "imagenet_sketch"
    # model_PATH: str = "/home/hyunsoo/inversion/DF_synthesis_LDM/classifier/imagenet-r_subset_by_domain/lr001_resnet50_p_T_imagenet-r_lpips_subset_sketch_0.944206008583691.pt"
    model_PATH: str = "/home/hyunsoo/inversion/DF_synthesis_LDM/classifier/imagenet-r_subset_by_domain/lr001_resnet50_p_T_imagenet-r_lpips_subset_art_0.9436619718309859.pt"
    category_path: str = "/home/hyunsoo/inversion/DF_synthesis_LDM/resnet_category.txt"
    # Affect training time
    early_stopping: int = 15
    num_train_epochs: int = 20

    # affect variability of the training images
    # i.e., also sets batch size with accumulation
    trainloader_size: int = 1 # 345
    number_of_prompts: int = 1  #3 how many different prompts to use
    batch_size: int = 1  # set to one due to gpu constraints
    gradient_accumulation_steps: int = 20  # same as the epoch size

    # Skip if there exists a token checkpoint
    skip_exists: bool = False

    # Train and Optimization
    lr: float = 0.0025 * trainloader_size
    betas: tuple = field(default_factory=lambda: (0.9, 0.999))
    weight_decay: float = 1e-2
    eps: float = 1e-08
    max_grad_norm: str = "1"
    seed: int = 35
    class_num: int = 10
    # Generative model
    guidance_scale: int = 15
    height: int = 512
    width: int = 512
    num_of_SD_inference_steps: int = 30

    # Discrimnative tokens
    domain_token: str = "dmtk"
    domain_initializer_token: str = "photo"

    # Path to save all outputs to
    output_path: Path = Path("results")
    save_as_full_pipeline = True

    # Cuda related
    device: str = "cuda"
    mixed_precision = "fp16"
    gradient_checkpointing = True

    # evaluate
    test_size: int = 9


def __post_init__(self):
    self.output_path.mkdir(exist_ok=True, parents=True)
