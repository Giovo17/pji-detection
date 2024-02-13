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


def export_models(models: Dict[str, Dict[str, str]], output_dir: str) -> None:
    """Export pytorch hub models."""

    for key, model_info in models.items():
        os.chdir(cache_path)
        model = torch.hub.load(model_info['repo'], model_info['model_name'], pretrained=True)
        os.chdir("../")

        
        # https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

        # https://it.mathworks.com/help/deeplearning/ref/importonnxnetwork.html
        

        #output_path = f'{output_dir}/{model_info['repo'].replace("/", "-")}_{model_info['model_name']}.pt'
        #torch.save(model.state_dict(), output_path)

    os.chdir("../")


def main():
    export_models(models, output_path)
    

if __name__ == "__main__":
    main()


