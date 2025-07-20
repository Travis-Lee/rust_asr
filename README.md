# build your docker 
docker build --no-cache --platform=linux/arm64 -t YourDockerAccountName/DockerName .

#How to run into a container
docker run -it \
  --name YourContainerName \
  -v $PWD/../rust_asr:/rust_asr \
  --net=host \
  -w /rust_asr \
  YourDockerAccountName/DockerName /bin/bash

#How to run the rust asr demo
cd onnx 
./bash/run.sh (convert pth model to onnx model)

cd data 
./get_hello_world_wave.py*

#build rust demo and execute demo inference
./run_asr.sh 

