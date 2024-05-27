from torch.utils.data import Dataset
import random
# from inet_classes import IDX2NAME as IDX2NAME_cls
# from inet_classes import CLS2IDX as CLS2IDX_cls

# imagenet_templates_small = ["A photo of {} from {}"]
imagenet_templates_small = ["A {} of {}"]
class PromptDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the promots for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        domain_token,
        prompt_suffix,
        tokenizer,
        trainloader_size,
        number_of_prompts,
        label_lst
    ):
        self.prompt_suffix = prompt_suffix
        self.tokenizer = tokenizer
        self.domain_token = domain_token
        self.trainloader_size = trainloader_size
        self.number_of_prompts = number_of_prompts
        self.label_lst = label_lst

    def __len__(self):
        return self.trainloader_size

    def __getitem__(self, index):
        example = {}
        # idx = random.randint(0,249)
        # suffix = self.label_lst[idx]
        # suffix = suffix.split(",")[0]
        text = imagenet_templates_small[index % self.number_of_prompts]
        text = text.format(self.domain_token, self.prompt_suffix)
        example["instance_prompt"] = text
        example["instance_prompt_ids"] = self.tokenizer(
            text,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        # example["instance_label"] = idx

        return example