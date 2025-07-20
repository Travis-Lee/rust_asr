use ort::{environment::Environment, session::SessionBuilder, tensor::OrtOwnedTensor, Value};
use std::sync::Arc;
use ndarray::{Array, Array3, Axis, CowArray, IxDyn};
use anyhow::Result;
use clap::Parser;

const LABELS: [&str; 29] = [
    "-", "|", "E", "T", "A", "O", "N", "I", "H", "S",
    "R", "D", "L", "U", "M", "W", "C", "F", "G", "Y",
    "P", "B", "V", "K", "'", "X", "J", "Q", "Z",
];

fn argmax(arr: &[f32]) -> usize {
    arr.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn remove_duplicates_and_blanks(indices: &[usize], blank: usize) -> Vec<usize> {
    let mut result = Vec::new();
    let mut prev = None;
    for &idx in indices {
        if Some(idx) != prev && idx != blank {
            result.push(idx);
        }
        prev = Some(idx);
    }
    result
}

// Read wav and convert to f32 tensor
fn load_wav_mono(path: &str) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = if spec.bits_per_sample == 16 {
        reader
            .into_samples::<i16>()
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect()
    } else {
        panic!("Only 16-bit PCM is supported");
    };

    // If it's multi-channel, simply take one channel
    Ok(samples)
}


#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// model path 
    #[arg(short, long, default_value = "asr.onnx")]
    model: Option<String>,
    /// audio path
    #[arg(short, long, default_value = "audio.wav")]
    audio: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    let model_path = match &args.model {
        Some(p) => p,
        None => {
            eprintln!("❌ Missing --model argument！");
            std::process::exit(1);
        }
    };
    
    if !std::path::Path::new(model_path).exists() {
        eprintln!("❌ Model file does not exist: {}", model_path);
        std::process::exit(1);
    }
    
    let audio_path = match &args.audio {
        Some(p) => p,
        None => {
            eprintln!("❌ Missing --audio argument！");
            std::process::exit(1);
        }
    };
    
    if !std::path::Path::new(audio_path).exists() {
        eprintln!("❌ Audio file does not exist: {}", audio_path);
        std::process::exit(1);
    }

    // Building Inference ENV
    let environment = Arc::new(Environment::builder().with_name("asr").build()?);
    let session = SessionBuilder::new(&environment)?
        .with_model_from_file(model_path)?;

    // Reading WAV Audio
    let audio_data = load_wav_mono(audio_path)?; 
    let audio_len = audio_data.len();

    let input_tensor: Array3<f32> = Array3::from_shape_vec((1, 1, audio_len), audio_data)?;
    let input_tensor = CowArray::from(input_tensor.into_dyn());

    let input_value = Value::from_array(session.allocator(), &input_tensor)?;
    let outputs: Vec<Value> = session.run(vec![input_value])?;

    let logits_tensor: OrtOwnedTensor<f32, IxDyn> = outputs[0].try_extract()?;
    let view = logits_tensor.view();
    let logits = view.index_axis(Axis(0), 0); // [T, C]

    // Decode into token ID sequence
    let pred_indices: Vec<usize> = logits
        .outer_iter()
        .map(|frame| argmax(frame.as_slice().unwrap()))
        .collect();

    println!("pred_indices: {:?}", pred_indices);
    let token_ids = remove_duplicates_and_blanks(&pred_indices, 0);
    println!("token_ids: {:?}", token_ids);
    let text: String = token_ids.iter().map(|&id| LABELS[id]).collect();
    println!("🗣️ Inference Text: {}", text);

    Ok(())
}



