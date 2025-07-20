#!/usr/bin/env python3
import torch
import torch
import torch.nn as nn 
import torchaudio
import onnx
from onnxsim import simplify
import argparse 
import os 

# 1.Define the same model structure as during training
class Wav2Vec2ASR(nn.Module):
    def __init__(self, encoder, hidden_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.decoder = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.squeeze(1)
        features, _ = self.encoder(x)
        logits = self.decoder(features)
        return logits

# 2. load pre-training encoder（must have original bundle/encoder）
def load_encoder():
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    encoder = bundle.get_model()
    labels = bundle.get_labels()
    print(f"labels:{labels}\n len:{len(labels)}")
    #print(f"bundle: {bundle}\nencoder: {encoder}\nlabels:{labels}")
    hidden_dim = None
    with torch.inference_mode():
        dummy_wave = torch.randn(1, 1, 16000)  # 1秒音频，batch=1
        features, _ = encoder(dummy_wave.squeeze(1))
        hidden_dim = features.shape[-1]

    num_classes = len(labels)
    return encoder, hidden_dim,num_classes

def pth2onnx(pmodel,xmodel):
    # 3. load model weights  
    encoder,hidden_dim,num_classes = load_encoder()
    asr_model = Wav2Vec2ASR(encoder, hidden_dim, num_classes)
    asr_model.load_state_dict(torch.load(pmodel, map_location="cpu"))
    asr_model.eval()

    #4.Preparing input data, must mach model input shape
    example_input = torch.randn(1, 1, 16000)  # batch=1, channel=1, length=16000

    # 5. Export onnx model
    torch.onnx.export(
        asr_model,
        example_input,
        xmodel,     #"asr_model.onnx",
        opset_version=17,
        input_names=["audio"],
        output_names=["logits"],
        dynamic_axes={
            "audio": {2: "sequence_length"},  # 第2维长度动态
            "logits": {1: "sequence_length"}  # 第1维长度动态
        }
    )
    print("ONNX model export success！")

def onnx2simplify(onnx_path):
    model=onnx.load(onnx_path)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, "final_asr.onnx")
    print("onnx model simplification success，save as final_asr.onnx")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser('Convert pth model to onnx model')

    parser.add_argument('-p', dest='pthDir', type=str, required=True, help='Path to pth model  directory')
    parser.add_argument('-x', dest='outName', type=str, default="asr.onnx", help='pth model convert to onnx model')
    parser.add_argument('-s', dest='xName', type=str, default="asr.onnx", help='final onnx model after onnx simplify')
   
    args = parser.parse_args()

    pth2onnx(args.pthDir,args.outName)

    onnx2simplify(args.xName)

