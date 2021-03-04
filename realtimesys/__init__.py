import pyaudio
import wave
from tqdm import tqdm
def record_audio(wave_out_path,record_second):
    CHUNK = 2048
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 48000
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    wf = wave.open(wave_out_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)

    wf2 = wave.open(r'2.wav', 'wb')
    wf2.setnchannels(2)
    wf2.setsampwidth(p.get_sample_size(FORMAT))
    wf2.setframerate(RATE)
    print("* recording")
    frames = []
    for i in tqdm(range(0, int(RATE / CHUNK * record_second))):
        data = stream.read(CHUNK)
        frames.append(data)
        wf.writeframes(data)
        # wf2.writeframes(data)
    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()


    d_o = b''.join(frames)
    wf2.writeframes(d_o)
    wf2.close()
    wf = wave.open(wave_out_path, 'rb')
    d = wf.readframes(wf.getnframes())
    print(d == d_o)

    import numpy as np
    data = np.frombuffer(d_o, dtype=np.int16)
    x = data.reshape(-1,2)
    import matplotlib.pyplot as plt
    plt.plot(data)
    plt.show()

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(x[:,0])
    plt.subplot(2,1,2)
    plt.plot(x[:,1])
    plt.show()

record_audio("output.wav",record_second=4)