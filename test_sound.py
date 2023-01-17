from sound import SoundPlayer, Sound
import logging
import time
import matplotlib.pyplot as plt

def sound_test(file_name):
    sound = Sound(file_name)
    data = sound.data[0]
    print(data.dtype)
    plt.plot(data[44100*4:44100*5])
    plt.show()
    duration = sound.metadata.nframes / float(sound.metadata.framerate)
    logging.info("Read %.2f sec sound." % (duration,))
    import pprint
    pprint.pprint(sound.metadata)

    position = [0]
    finished = [False]

    def _make_samples(n_frames):
        endpoint = position[0] + n_frames
        if endpoint > data.size:
            endpoint = data.size
            logging.info("Sound finished.")
            finished[0] = True
        samples = data[position[0]:endpoint]
        position[0] = endpoint
        bytes = sound.encode_samples(samples)
        return bytes

    #player = SoundPlayer.from_sound(sound, _make_samples, frames_per_buffer=4096)
    #player.start()
    p = pyaudio.PyAudio()

    while not finished[0]:
        time.sleep(.1)
    player.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    filename = "Mimus_longicaudatus_-_Long_tailed_Mockingbird.wav"
    sound_test(filename)
    print("Test complete.")
