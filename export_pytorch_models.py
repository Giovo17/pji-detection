import os
import torch
from typing import Dict


output_path = os.path.join(os.getcwd(), "models") 

models = {
    'yolov5': {
        'repo': 'ultralytics/yolov5',
        'model_name': 'yolov5s'
    }
}


def export_models(models: Dict[str, Dict[str, str]], output_path: str) -> None:
    """Export pytorch hub models."""

    for key, model_info in models.items():
        model = torch.hub.load(model_info['repo'], model_info['model_name'], pretrained=True)

        torch.save(model.state_dict(), f'{output_path}/{model_info['repo'].replace("/", "-")}_{model_info['model_name']}.pt')


def main():
    export_models(models, output_path)
    

if __name__ == "__main__":
    main()


