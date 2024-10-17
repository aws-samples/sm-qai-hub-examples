import os
import uuid
from tqdm import tqdm
import yaml
from datasets import load_dataset


class CreateYoloHFDataset:
    """
    A class to create a YOLO format dataset from a Hugging Face dataset.

    Attributes:
        hf_dataset_name (str): Name of the Hugging Face dataset.
        labels_names (list): List of label names.
    """

    def __init__(self, hf_dataset_name: str, labels_names: list):
        """
        Initializes the CreateYoloHFDataset with the dataset name and label names.

        Args:
            hf_dataset_name (str): The Hugging Face dataset name.
            labels_names (list): A list of label names.
        """
        self.hf_dataset_name = hf_dataset_name
        self.loaded_dataset = load_dataset(self.hf_dataset_name, "full")
        self.splits = list(self.loaded_dataset.keys())
        print(f"Loaded dataset: {self.hf_dataset_name} ✅")
        self.data_paths = {}
        self.data_yaml = {}
        self.labels = labels_names

    def build_data_yaml(self, data_write_path: str):
        """
        Builds the data.yaml file necessary for YOLO training.

        Args:
            data_write_path (str): Path where the data.yaml file will be saved.
        """
        self.data_yaml = {"path": data_write_path}

        for split_i in self.data_paths:
            self.data_yaml[split_i] = "/".join(self.data_paths[split_i]["images"].split('/')[-2:])
        
        labels_dict = {idx: label for idx, label in enumerate(self.labels)}
        self.data_yaml["names"] = labels_dict
        
        _yaml_path = os.path.join(data_write_path, "data_local.yaml")
        with open(_yaml_path, 'w') as file:
            yaml.dump(self.data_yaml, file, default_flow_style=False)
        
        self.data_yaml["data_yaml"] = _yaml_path

    def create_local_dataset(self, data_write_path: str):
        """
        Creates a local dataset in YOLO format by saving images and labels locally.

        Args:
            data_write_path (str): The base directory where images and labels will be saved.
        """
        for split_i in self.splits:
            split_dataset = self.loaded_dataset[split_i]
            self.data_paths[split_i] = {
                "images": os.path.join(data_write_path, "images", split_i),
                "labels": os.path.join(data_write_path, "labels", split_i)
            }
            os.makedirs(self.data_paths[split_i]["images"], exist_ok=True)
            os.makedirs(self.data_paths[split_i]["labels"], exist_ok=True)

            for ix, row in tqdm(enumerate(split_dataset), total=len(split_dataset), desc=f"{split_i}"):
                image_i = row["image"] 
                bboxes = row["objects"]["bbox"]
                temp_name = str(uuid.uuid4())
                img_name = f"{temp_name}.jpg"
                label_name = f"{temp_name}.txt"
                
                if bboxes:
                    category = row["objects"]["category"]
                    width, height = row["width"], row["height"]
                    bboxes_str = ""
                    total_labels = len(bboxes) - 1

                    for i, (bbox, cat) in enumerate(zip(bboxes, category)):
                        x_c, y_c, w, h = bbox
                        x_c, y_c, w, h = (
                            str(round(x_c / width, 3)),
                            str(round(y_c / height, 3)),
                            str(round(w / width, 3)),
                            str(round(h / height, 3))
                        )
                        bboxes_str += " ".join([str(cat), x_c, y_c, w, h])
                        if i < total_labels:
                            bboxes_str += "\n"

                    # Save image and label
                    image_i.save(os.path.join(self.data_paths[split_i]["images"], img_name))
                    with open(os.path.join(self.data_paths[split_i]["labels"], label_name), "w") as label_file:
                        label_file.write(bboxes_str)
        
        # Build the data.yaml file after creating the dataset
        self.build_data_yaml(data_write_path=data_write_path)
