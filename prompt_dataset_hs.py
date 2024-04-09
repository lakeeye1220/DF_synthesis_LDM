from torch.utils.data import Dataset
import random
from inet_classes import IDX2NAME as IDX2NAME_INET

imagenet_templates_small = ["A photo of {} from {}"]

class PromptDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the promots for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        model_placeholder_token,
        # class_placeholder_token,
        # prompt_suffix,
        tokenizer,
        epoch_size,
        number_of_prompts,
    ):
        self.tokenizer = tokenizer
        self.model_placeholder_token = model_placeholder_token
        # self.class_placeholder_token = class_placeholder_token
        self.epoch_size = epoch_size
        self.number_of_prompts = number_of_prompts

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, index):
        example = {}
        rand_idx = random.randint(0,999)
        suffix = IDX2NAME_INET[rand_idx]
        suffix = suffix.split(",")[0]
        text = imagenet_templates_small[index % self.number_of_prompts]
        text = text.format(suffix, self.model_placeholder_token)
        example["instance_prompt"] = text
        example["instance_prompt_ids"] = self.tokenizer(
            text,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        example["instance_label"] = rand_idx

        return example
