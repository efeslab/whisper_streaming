#!/usr/bin/env python3
from whisper_online import *

import sys
import argparse
import os
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

# server options
parser.add_argument("--host", type=str, default='localhost')
parser.add_argument("--port", type=int, default=43007)
parser.add_argument("--device", type=str, default="gpu", choices=["cpu","gpu"], help="Device to use for inference. Default is gpu. If you want to use CPU, set it to cpu.")
parser.add_argument("--warmup-file", type=str, dest="warmup_file", 
        help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .")

# options from whisper_online
add_shared_args(parser)
args = parser.parse_args()

set_logging(args,logger,other="")

# setting whisper object by args 

SAMPLING_RATE = 16000

size = args.model
language = args.lan
asr, online = asr_factory(args)
min_chunk = args.min_chunk_size

# warm up the ASR because the very first transcribe takes more time than the others. 
# Test results in https://github.com/ufal/whisper_streaming/pull/81
msg = "Whisper is not warmed up. The first chunk processing may take longer."
if args.warmup_file:
    if os.path.isfile(args.warmup_file):
        a = load_audio_chunk(args.warmup_file,0,1)
        asr.transcribe(a)
        print("Whisper is warmed up.")
    else:
        logger.critical("The warm up file is not available. "+msg)
        sys.exit(1)
else:
    logger.warning(msg)


######### Server objects

import line_packet
import socket

class Connection:
    '''it wraps conn object'''
    # do 2 second
    PACKET_SIZE = 2*SAMPLING_RATE*2 # 2 seconds of audio, 16 bit, mono

    def __init__(self, conn):
        self.conn = conn
        self.last_line = ""

        self.conn.setblocking(True)

    def send(self, line):
        '''it doesn't send the same line twice, because it was problematic in online-text-flow-events'''
        # if line == self.last_line:
        #     return
        line_packet.send_one_line(self.conn, line)
        self.last_line = line
        

    def receive_lines(self):
        in_line = line_packet.receive_lines(self.conn)
        return in_line

    # def non_blocking_receive_audio(self):
    #     try:
    #         r = self.conn.recv(self.PACKET_SIZE)
    #         return r
    #     except ConnectionResetError:
    #         return None

    def non_blocking_receive_audio(self):
        """
        Receives audio data, handling cases where TCP may split a single chunk into multiple packets.
        Returns complete chunks of PACKET_SIZE bytes, or None if connection closed.
        """
        try:
            # Initialize buffer for collecting fragmented data
            buffer = b''
            bytes_needed = self.PACKET_SIZE
            
            # Keep receiving until we have a complete packet
            while len(buffer) < self.PACKET_SIZE:
                # Try to receive remaining bytes needed
                fragment = self.conn.recv(bytes_needed)
                
                # If we get empty data, the connection is closed
                if not fragment:
                    if not buffer:  # If buffer is also empty, return None
                        return None
                    else:  # Otherwise return what we have with a warning
                        print(f"WARNING: Received incomplete audio chunk: {len(buffer)} bytes instead of {self.PACKET_SIZE}")
                        return buffer
                
                # Add the fragment to our buffer
                buffer += fragment
                bytes_needed = self.PACKET_SIZE - len(buffer)
                
                print(f"Received fragment: {len(fragment)} bytes, total so far: {len(buffer)}/{self.PACKET_SIZE}")
                
            print(f"Completed full audio chunk: {len(buffer)} bytes")
            return buffer
            
        except ConnectionResetError:
            return None
        except socket.error as e:
            print(f"Socket error: {e}")
            return None


import io
import soundfile

# wraps socket and ASR object, and serves one client connection. 
# next client should be served by a new instance of this object
class ServerProcessor:

    def __init__(self, c, online_asr_proc, min_chunk):
        self.connection = c
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk

        self.last_end = None

        self.is_first = True
        self.chunk_num = 0

    def receive_audio_chunk(self):
        # receive all audio that is available by this time
        # blocks operation if less than self.min_chunk seconds is available
        # unblocks if connection is closed or a chunk is available
        out = []
        minlimit = self.min_chunk*SAMPLING_RATE
        # while sum(len(x) for x in out) < minlimit:
        raw_bytes = self.connection.non_blocking_receive_audio()
        if not raw_bytes:
            # break
            return None
        print("received audio:",len(raw_bytes), "bytes", raw_bytes[:10])
        sf = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1,endian="LITTLE",samplerate=SAMPLING_RATE, subtype="PCM_16",format="RAW")
        audio, _ = librosa.load(sf,sr=SAMPLING_RATE,dtype=np.float32)
        out.append(audio)
        if not out:
            return None
        conc = np.concatenate(out)
        if self.is_first and len(conc) < minlimit:
            return None
        self.is_first = False
        self.chunk_num += 1
        print(f"[{self.chunk_num}]: received audio")
        return np.concatenate(out)

    def format_output_transcript(self,o):
        # output format in stdout is like:
        # 0 1720 Takhle to je
        # - the first two words are:
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript

        # This function differs from whisper_online.output_transcript in the following:
        # succeeding [beg,end] intervals are not overlapping because ELITR protocol (implemented in online-text-flow events) requires it.
        # Therefore, beg, is max of previous end and current beg outputed by Whisper.
        # Usually it differs negligibly, by appx 20 ms.

        if o[0] is not None:
            beg, end = o[0]*1000,o[1]*1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)

            self.last_end = end
            print("%1.0f %1.0f %s" % (beg,end,o[2]),flush=True,file=sys.stderr)
            return "%1.0f %1.0f %s" % (beg,end,o[2])
        else:
            logger.debug("No text in this segment")
            print("No input",flush=True,file=sys.stderr)
            # return f"No input {time.time()}"
            return "No input"
            # return None

    # def send_result(self, o):
    #     msg = self.format_output_transcript(o)
    #     print(f"send message: {msg}")
        
    #     if msg is not None:
    #         # Encode to bytes and send all
    #         try:
    #             self.connection.sendall((msg + "\n").encode('utf-8'))
    #         except Exception as e:
    #             print(f"Failed to send message: {e}")
    def send_result(self, o):
        msg = self.format_output_transcript(o)
        # remove all new lines
        msg = msg.replace("\n", " ")
        print(f"send message: {msg}")
        if msg is not None:
            self.connection.send(msg)

    def process(self):
        # handle one client connection
        self.online_asr_proc.init()
        while True:
            a = self.receive_audio_chunk()
            if a is None:
                break
            self.online_asr_proc.insert_audio_chunk(a)
            start_time = time.time()
            o = online.process_iter()
            end_time = time.time()
            print(f"[{self.chunk_num}]: processing time: {end_time - start_time:.6f} seconds, start time: {start_time:.6f} seconds, end time: {end_time:.6f} seconds")
            try:
                self.send_result(o)
            except Exception as e:
                logger.error(f"Error sending result: {e}")
                break

#        o = online.finish()  # this should be working
#        self.send_result(o)



# server loop

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((args.host, args.port))
    s.listen(1)
    print('Listening on'+str((args.host, args.port)))
    while True:
        conn, addr = s.accept()
        print('Connected to client on {}'.format(addr))
        connection = Connection(conn)
        proc = ServerProcessor(connection, online, args.min_chunk_size)
        proc.process()
        conn.close()
        print('Connection to client closed')
print('Connection closed, terminating.')
