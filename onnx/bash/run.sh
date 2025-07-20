./pth2onnx.py -p ../model/asr_model.pt -x asr.onnx -s asr.onnx
cp -rf final_asr.onnx ../model/asr.onnx
rm -rf *onnx
