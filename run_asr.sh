cargo clean 

cargo build --release 

./target/release/rust_asr --model ./model/asr.onnx --audio ./data/hello_world_linux.wav
