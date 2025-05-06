#!/usr/bin/env python3
import sys
import time
import socket
import subprocess
import argparse
import os
import select
from collections import deque

def send_audio_in_chunks(audio_file_path, host, port, chunk_seconds=2.0):
    """
    Converts audio file to raw PCM using ffmpeg and sends it to a server in timed chunks.
    Receives responses after each chunk is sent and measures processing time.
    
    Args:
        audio_file_path: Path to the audio file
        host: Server hostname or IP
        port: Server port
        chunk_seconds: Size of each chunk in seconds
    """
    # Get audio duration and info using ffprobe
    duration_cmd = [
        'ffprobe', 
        '-v', 'error', 
        '-show_entries', 'format=duration', 
        '-of', 'default=noprint_wrappers=1:nokey=1', 
        audio_file_path
    ]
    
    try:
        duration = float(subprocess.check_output(duration_cmd).decode('utf-8').strip())
        print(f"Audio duration: {duration:.2f} seconds")
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return
    
    # Calculate bytes per second for 16-bit, 16kHz, mono audio
    bytes_per_second = 16000 * 2  # Sample rate * bytes per sample
    chunk_size = int(bytes_per_second * chunk_seconds)
    
    # Set up ffmpeg command to convert audio to raw PCM
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', audio_file_path,
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        '-'
    ]
    
    # Connect to server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Increase TCP send buffer size (e.g., to 1MB or more)
    # send_buffer_size = 1024 * 1024 * 1024  # 5MB
    # sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, send_buffer_size)

    # Check what was actually set
    # actual_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    # print(f"Requested buffer size: {send_buffer_size} bytes")
    # print(f"Actual buffer size: {actual_size} bytes")
    
    # FIFO queue to track timestamps for sent chunks
    chunk_timestamps = deque()
    
    try:
        sock.connect((host, port))
        print(f"Connected to {host}:{port}")
        
        # Make socket non-blocking for receiving responses without waiting
        sock.setblocking(0)
        
        # Start ffmpeg process
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        total_bytes_sent = 0
        chunk_count = 0
        
        # Send data in chunks
        while True:
            # Read chunk of PCM data
            data = ffmpeg_process.stdout.read(chunk_size)
            
            # If no more data, break
            if not data:
                break
            
            # Send the data
            send_time = time.time()
            sock.sendall(data)
            
            # Record the chunk number and timestamp in the queue
            chunk_count += 1
            chunk_timestamps.append((chunk_count, send_time))
            
            total_bytes_sent += len(data)
            
            chunk_duration = len(data) / bytes_per_second
            print(f"Sent chunk {chunk_count}: {len(data)} bytes ({chunk_duration:.2f} seconds) at {send_time:.6f}")

            # Check if this is a partial final chunk that needs padding
            if len(data) < chunk_size and chunk_duration < chunk_seconds:
                # Calculate how many bytes of silence to add
                silence_bytes_needed = chunk_size - len(data)
                # Create silence padding (zeros for PCM)
                silence_padding = b'\x00' * silence_bytes_needed
                print(f"Padding final chunk with {silence_bytes_needed} bytes of silence")
                sock.sendall(silence_padding)
                total_bytes_sent += silence_bytes_needed
            
            # Try to receive response immediately after sending each chunk
            try_receive_response(sock, chunk_timestamps)
                
            # Sleep to simulate real-time streaming
            # Only if there's more data to send
            if ffmpeg_process.stdout.peek():
                # While sleeping, continue checking for responses
                start_time = time.time()
                while time.time() - start_time < chunk_seconds:
                    try_receive_response(sock, chunk_timestamps)
                    time.sleep(0.1)  # Short sleep to avoid CPU spinning
            
        print(f"Finished sending {chunk_count} chunks ({total_bytes_sent} bytes)")
        
        # Wait a bit more for any final responses
        wait_for_final_responses(sock, chunk_timestamps)
        
        # Report any chunks that never received responses
        if chunk_timestamps:
            print("\nChunks without responses:")
            for chunk_num, send_time in chunk_timestamps:
                elapsed = time.time() - send_time
                print(f"  Chunk {chunk_num}: Sent {elapsed:.2f} seconds ago, no response received")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        if 'ffmpeg_process' in locals() and ffmpeg_process:
            ffmpeg_process.terminate()
        sock.close()
        print("Connection closed")

# def try_receive_response(sock, chunk_timestamps):
#     if not chunk_timestamps:
#         return
    
#     try:
#         response = sock.recv(4096)
#         if not response:
#             return
        
#         # Split by newline to separate messages
#         messages = response.decode('utf-8', errors='replace').strip().split('\n')
        
#         for message in messages:
#             if message and chunk_timestamps:
#                 chunk_num, send_time = chunk_timestamps.popleft()
#                 receive_time = time.time()
#                 processing_time = receive_time - send_time
                
#                 print(f"\nResponse for chunk {chunk_num}:")
#                 print(f"  Processing time: {processing_time:.6f} seconds")
#                 print(f"  Content: {message}")
                
#     except Exception as e:
#         print(f"Error receiving data: {e}")

def try_receive_response(sock, chunk_timestamps):
    """
    Attempt to receive response from socket without blocking
    
    Args:
        sock: Socket to receive from
        chunk_timestamps: FIFO queue of (chunk_number, timestamp) tuples
    """
    # Skip if there are no chunks waiting for responses
    if not chunk_timestamps:
        return
    
    # Use select to check if there's data available to read
    # sock.settimeout(0.5)  # 500ms timeout

    readable, _, _ = select.select([sock], [], [], 0)
    if readable:
        try:
            # There's data available to read
            response = sock.recv(4096)
            if response:
                # Split by newline to separate messages
                # print how many bytes received
                print(f"Received {len(response)} bytes of response data")
                messages = response.decode('utf-8', errors='replace').strip().split('\n')
                
                for message in messages:
                    if message and chunk_timestamps:  # Skip empty messages
                        # Get the oldest chunk's info
                        chunk_num, send_time = chunk_timestamps.popleft()
                        receive_time = time.time()
                        processing_time = receive_time - send_time
                        
                        # Display the response with timing information
                        print(f"\nResponse for chunk {chunk_num}:")
                        print(f"  Processing time: {processing_time:.6f} seconds")
                        print(f"  Content: {message}")
                        print(f"  Received at: {receive_time:.6f}")
                        print(f"  Sent at: {send_time:.6f}")
                    
        except Exception as e:
            print(f"Error receiving data: {e}")

def check_for_additional_responses(sock, chunk_timestamps):
    """Check for additional responses without delay"""
    while chunk_timestamps:
        # Check if more data is available immediately
        readable, _, _ = select.select([sock], [], [], 0)
        if not readable:
            break
            
        try:
            response = sock.recv(4096)
            if not response:
                break
                
            # Get the oldest chunk's info
            chunk_num, send_time = chunk_timestamps.popleft()
            receive_time = time.time()
            processing_time = receive_time - send_time
            
            # Display the response with timing information
            print(f"\nResponse for chunk {chunk_num}:")
            print(f"  Processing time: {processing_time:.6f} seconds")
            print(f"  Content: {response.decode('utf-8', errors='replace')}")
        except Exception as e:
            print(f"Error receiving additional data: {e}")
            break

def wait_for_final_responses(sock, chunk_timestamps, timeout=500000000):
    """
    Wait for any final responses from the server
    
    Args:
        sock: Socket to receive from
        chunk_timestamps: FIFO queue of (chunk_number, timestamp) tuples
        timeout: Maximum time to wait in seconds
    """
    if not chunk_timestamps:
        return
        
    print(f"Waiting up to {timeout} seconds for responses to {len(chunk_timestamps)} remaining chunks...")
    end_time = time.time() + timeout
    
    while time.time() < end_time and chunk_timestamps:
        try_receive_response(sock, chunk_timestamps)
        time.sleep(0.1)  # Short sleep to avoid CPU spinning

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stream audio to server in real-time chunks')
    parser.add_argument('audio_file', help='Path to audio file (any format ffmpeg supports)')
    parser.add_argument('--host', default='localhost', help='Server hostname or IP')
    parser.add_argument('--port', type=int, default=43001, help='Server port')
    parser.add_argument('--interval', type=float, default=2.0, help='Chunk interval in seconds')
    
    args = parser.parse_args()
    
    send_audio_in_chunks(args.audio_file, args.host, args.port, args.interval)