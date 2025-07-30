import numpy as np
import soundfile as sf
from datasets import load_dataset
import os

repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def numpy_to_wav(audio_data, output_filename, sampling_rate=16000):
    """
    Convert NumPy array to WAV file.
    
    Parameters:
    -----------
    audio_data : numpy.ndarray
        Audio data as NumPy array
    output_filename : str
        Output WAV filename
    sampling_rate : int, optional
        Sampling rate of the audio, by default 16000
    """
    # Ensure the array is float32 for best compatibility
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # Write the NumPy array directly to a WAV file
    sf.write(output_filename, audio_data, sampling_rate)
    print(f"WAV file saved as {output_filename}")

# Example usage
def main():
    # Load the dataset (similar to your original code)
    livecaptions_prompts = []
    
    def load_livecaptions_dataset():
        """Load the live captions dataset"""
        ds_livecaptions = load_dataset("distil-whisper/earnings21")
        ds_livecaptions = ds_livecaptions["test"]
        for item in ds_livecaptions:
            livecaptions_prompts.append(item['audio'])
            break
        return len(ds_livecaptions)
    
    # Load the dataset
    total_items = load_livecaptions_dataset()
    print(f"Loaded {total_items} audio files from dataset")
    
    # Process all files
    output_dir = f"{repo_dir}/applications/LiveCaptions/whisper-earnings21"
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert each numpy array to a WAV file
    for i, audio_item in enumerate(livecaptions_prompts):
        audio_array = audio_item['array']
        sampling_rate = audio_item['sampling_rate']
        original_path = audio_item['path']
        
        # Extract filename from path (without extension)
        filename = os.path.splitext(os.path.basename(original_path))[0]
        output_filename = os.path.join(output_dir, f"{filename}.wav")
        
        print(f"Converting file {i+1}/{len(livecaptions_prompts)}: {original_path}")
        numpy_to_wav(audio_array, output_filename, sampling_rate)

if __name__ == "__main__":
    main()
