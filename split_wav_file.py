import os
import wave
import argparse
import numpy as np
from pydub import AudioSegment

def get_duration(wav_file_path):
    """Get the duration of a WAV file in seconds."""
    with wave.open(wav_file_path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
        return duration

def split_wav_file(input_wav, output_dir, chunk_length=60):
    """
    Split a WAV file into chunks of specified length.
    
    Parameters:
    -----------
    input_wav : str
        Path to the input WAV file
    output_dir : str
        Directory to save the chunks
    chunk_length : int, optional
        Length of each chunk in seconds, default is 60 (1 minute)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get filename without extension
    base_filename = os.path.splitext(os.path.basename(input_wav))[0]
    
    # Load the audio file
    print(f"Loading audio file: {input_wav}")
    audio = AudioSegment.from_wav(input_wav)
    
    # Get total duration in milliseconds
    total_duration_ms = len(audio)
    total_duration_sec = total_duration_ms / 1000
    
    # Calculate number of chunks
    chunk_length_ms = chunk_length * 1000
    num_chunks = int(np.ceil(total_duration_ms / chunk_length_ms))
    
    print(f"Audio duration: {total_duration_sec:.2f} seconds ({total_duration_sec/60:.2f} minutes)")
    print(f"Splitting into {num_chunks} chunks of {chunk_length} seconds each")
    
    # Split and save chunks
    for i in range(num_chunks):
        start_ms = i * chunk_length_ms
        end_ms = min((i + 1) * chunk_length_ms, total_duration_ms)
        
        # Extract chunk
        chunk = audio[start_ms:end_ms]
        
        # Generate output filename
        output_filename = f"{base_filename}_chunk_{i+1:03d}.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save chunk
        chunk.export(output_path, format="wav")
        
        # Calculate chunk duration
        chunk_duration = (end_ms - start_ms) / 1000
        
        print(f"Saved chunk {i+1}/{num_chunks}: {output_filename} ({chunk_duration:.2f} seconds)")

def main():
    parser = argparse.ArgumentParser(description='Split a WAV file into chunks of specified length')
    parser.add_argument('--input_file', help='Path to input WAV file')
    parser.add_argument('--output-dir', default='chunks', help='Directory to save chunks')
    parser.add_argument('--chunk-length', type=int, default=60, help='Length of each chunk in seconds')
    
    args = parser.parse_args()
    
    # Verify input file exists and is a WAV file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        return
    
    if not args.input_file.lower().endswith('.wav'):
        print(f"Error: Input file {args.input_file} is not a WAV file")
        return
    
    # Split the WAV file
    split_wav_file(args.input_file, args.output_dir, args.chunk_length)
    print(f"All chunks saved to {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()