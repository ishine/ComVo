import os
import torch
import torchaudio
import argparse
import yaml
import importlib


def load_specific_module(model, state_dict, module_name):
    pretrained_dict = {
        k.replace(f"{module_name}.", "", 1): v
        for k, v in state_dict.items()
        if k.startswith(f"{module_name}.")
    }

    if len(pretrained_dict) == 0:
        raise ValueError(
            f"No parameters found for module '{module_name}' in checkpoint"
        )

    model.load_state_dict(pretrained_dict, strict=True)
    model.eval()


def build_module(module_cfg):
    class_path = module_cfg["class_path"]
    pkg = ".".join(class_path.split(".")[:-1])
    name = class_path.split(".")[-1]
    cls = getattr(importlib.import_module(pkg), name)
    return cls(**module_cfg["init_args"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--wavfile", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./results")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, "r"))

    feature_extractor = build_module(config["model"]["init_args"]["feature_extractor"])
    backbone = build_module(config["model"]["init_args"]["backbone"])
    head = build_module(config["model"]["init_args"]["head"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    if "state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'state_dict'")
    state_dict = checkpoint["state_dict"]

    load_specific_module(feature_extractor, state_dict, "feature_extractor")
    load_specific_module(backbone, state_dict, "backbone")
    load_specific_module(head, state_dict, "head")

    feature_extractor = feature_extractor.to(device)
    backbone = backbone.to(device)
    head = head.to(device)

    target_sr = config["model"]["init_args"]["sample_rate"]

    os.makedirs(args.out_dir, exist_ok=True)

    audio_path = args.wavfile
    basename = os.path.basename(audio_path)
    save_path = os.path.join(args.out_dir, basename)

    with torch.no_grad():
        y, sr = torchaudio.load(audio_path)

        if y.size(0) > 1:
            y = y.mean(dim=0, keepdim=True)

        if sr != target_sr:
            y = torchaudio.functional.resample(y, sr, target_sr)
            sr = target_sr

        y = y.to(device)

        features = feature_extractor(y)
        x = backbone(features)
        audio_output = head(x).cpu()

        torchaudio.save(save_path, audio_output, sr)

    print(save_path)
