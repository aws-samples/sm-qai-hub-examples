import os
import shutil
import argparse
import logging
from datetime import datetime
import numpy as np
import random
import pprint
import torch
from ultralytics import YOLO


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
    
set_seed()

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler and set level to info
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add formatter to console handler
console_handler.setFormatter(formatter)

# Add console handler to logger
logger.addHandler(console_handler)

def parse_args():
    # Parse input arguments
    parser = argparse.ArgumentParser(description="YOLOv8 Training Parameters")
    
    # model size
    parser.add_argument('--yolov8-model', type=str, required=True, help='Which Yolo V8 model to train')
    parser.add_argument('--dataset-yaml', type=str,  default=os.environ["SM_CHANNEL_TRAINING"], help='Path to Yolo V8 data.yaml')
    parser.add_argument("--model-save-dir", type=str, default=os.environ["SM_MODEL_DIR"], help='Local sagemaker training path to store trained model')
    parser.add_argument("--artifacts-output-dir", type=str, default=os.environ["SM_OUTPUT_DIR"], help='Path to save runs and other local training artifacts')
    
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--num-train-loops', type=int, default=10, help='Number of training loops')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for training')
    parser.add_argument('--layer-freeze', type=int, default=0, help='Number of layers to freeze during training')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    
    # Data augmentations
    parser.add_argument('--hsv-h', type=float, default=0.1, help='Hue adjustment amount')
    parser.add_argument('--hsv-s', type=float, default=0.7, help='Saturation adjustment amount')
    parser.add_argument('--hsv-v', type=float, default=0.4, help='Value adjustment amount')
    parser.add_argument('--degrees', type=float, default=0.4, help='Rotation degree for augmentation')
    parser.add_argument('--translate', type=float, default=0.3, help='Translate factor for augmentation')
    parser.add_argument('--scale', type=float, default=0.5, help='Scaling factor for augmentation')
    parser.add_argument('--shear', type=float, default=0.01, help='Shear factor for augmentation')
    parser.add_argument('--perspective', type=float, default=0.001, help='Perspective adjustment for augmentation')
    parser.add_argument('--flipud', type=float, default=0.3, help='Flip upside down probability')
    parser.add_argument('--fliplr', type=float, default=0.3, help='Flip left-right probability')
    parser.add_argument('--bgr', type=float, default=0.1, help='BGR to RGB conversion probability')
    parser.add_argument('--mosaic', type=float, default=0.5, help='Mosaic augmentation probability')
    parser.add_argument('--mixup', type=float, default=0.5, help='MixUp augmentation probability')
    parser.add_argument('--copy-paste', type=float, default=0.4, help='Copy-paste augmentation probability')
    parser.add_argument('--erasing', type=float, default=0.2, help='Random erasing augmentation probability')
    parser.add_argument('--crop-fraction', type=float, default=0.1, help='Fraction to crop during augmentation')

    return parser.parse_args()


def main():
    
    logger.info("Starting YOLO V8 model training!")
    # Parse the arguments
    args = parse_args()
    logger.info(f"Runtime arguments: {vars(args)}")

    # Load the YOLO model and pin to GPU
    logger.info(f"Downloading Yolo Model Architecture {args.yolov8_model}")
    logger.info(f"------------>>> {os.listdir()}")
    model = YOLO(args.yolov8_model)
    _ = model.to('cuda')  # Pin model to GPU
    logger.info(f"\n============ MODEL ============\n{model}")

    # Training or inference can be set up here using the parsed args
    print("REMOVE", os.listdir(os.path.dirname(args.dataset_yaml)))
    if not args.dataset_yaml.endswith('yaml'):
        dataset_yaml = os.path.join(args.dataset_yaml, "data.yaml")
    else:
        dataset_yaml = args.dataset_yaml
    logger.info(f"Running model tuning on dataset: {dataset_yaml}")
    
    tuned_model = model.train(
        data=dataset_yaml,
        batch=args.batch_size,
        imgsz=args.img_size,
        epochs=args.epochs,
        freeze=args.layer_freeze,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        # bgr=args.bgr,  # Uncomment this if the model supports BGR as a parameter
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        erasing=args.erasing,
        crop_fraction=args.crop_fraction,
        save_dir=args.artifacts_output_dir
    )
    
    # Run inference
    logger.info("\nrunning model evaluation...")
    metrics = model.val(data=dataset_yaml)
    
    print("\n================= Validation Confusion Matrix =================")
    print(metrics.confusion_matrix.matrix)
    print("\n")
    
    logger.info("\nrunning model evaluation...")
    model.save(f"{args.model_save_dir}/{args.yolov8_model.split('.')[0]}_fine_tuned.pt")
    shutil.copy2("runs/detect/train/weights/best.pt", f"{args.model_save_dir}/best.pt")

if __name__ == '__main__':
    main()
