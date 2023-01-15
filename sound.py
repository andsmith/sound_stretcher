import numpy as np
import sys
import os
import subprocess
import tempfile
import logging

import wave
import time
import pyaudio


class SoundPlayer(object):
    def __init__(self, sample_width, frame_rate, channels, sample_generator):
        """
        Open stream for playing.
        :param sample_width:  bytes per frame
        :param channels: 1 or 2
        :param rate:  frame rate (e.g. 44100)
        """
        self._sample_width = sample_width
        self._frame_rate = frame_rate
        self._channels = channels
        self._sample_gen = sample_generator
        self._p = pyaudio.PyAudio()
        self._stream = None

    def start_playback(self):
        self._stream = self._p.open(format=self._p.get_format_from_width(self._sample_width),
                                    channels=self._channels,
                                    rate=self._frame_rate,
                                    output=True,
                                    stream_callback=self._get_samples)

    def _get_samples(self, in_data, frame_count, time_info, status):
        data = self._sample_gen(frame_count)
        # If len(data) is less than requested frame_count, PyAudio automatically
        # assumes the stream is finished, and the stream stops.

        code = pyaudio.paContinue
        if len(data) < frame_count:
            self._stream = False
            code = pyaudio.paComplete

        return data, code

    def stop_playback(self):
        self._stream.close()
        self._stream = None

    def shutdown(self):
        self._p.terminate()


EXTENSIONS = ['.m4a', '.ogg', '.mp3']


def read_sound(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.wav':
        return _read_wav(filename)
    elif ext in EXTENSIONS:
        return _read_other(filename)
    else:
        raise Exception("unknown file type, not one of %s:  %s" % (EXTENSIONS, ext))


def _read_wav(filename):
    with wave.open(filename, 'rb') as wav:
        wav_params = wav.getparams()
        data = wav.readframes(wav_params.nframes)
    data = _convert_from_bytes(data, wav_params)
    duration = wav_params.nframes / float(wav_params.framerate)
    logging.info("Read file:  %s (%.4f sec, %i Hz, %i channel(s))" % (filename, duration,
                                                                      wav_params.framerate,
                                                                      wav_params.nchannels))
    return data, wav_params


def _read_other(filename):
    with tempfile.TemporaryDirectory() as temp_dir:
        in_stem = os.path.splitext(filename)[0]
        temp_wav = os.path.join(temp_dir, "%s.wav" % (in_stem,))
        print("Converting:  %s  -->  %s" % (filename, temp_wav))
        cmd = ['ffmpeg', '-i', filename, temp_wav]
        print("Running:  %s" % (" ".join(cmd)))
        _ = subprocess.run(cmd, capture_output=True)
        return _read_wav(temp_wav)


def _convert_from_bytes(data, wav_params):
    # figure out data type
    if wav_params.sampwidth == 1:
        n_data = np.frombuffer(data, dtype=np.uint8)
    elif wav_params.sampwidth == 2:
        n_data = np.frombuffer(data, dtype=np.int16)
    elif wav_params.sampwidth == 4:
        n_data = np.frombuffer(data, dtype=np.int32)
    else:
        raise Exception("Unknown sample width:  %i bytes" % (wav_params.samplewidth,))

    # separate interleaved channel data
    n_data = [n_data[offset::wav_params.nchannels] for offset in range(wav_params.nchannels)]

    return n_data


'''

class StretchyWave(object):

    def save_slowed(self, factor):

        out_name = "%s_slowed_%.2f.wav" % (self._in_stem, factor)
        if os.path.exists(out_name):
            print("OVERWRITING:  %s" % (out_name,))
        else:
            print("Saving to:  %s" % (out_name,))
        time_indices = np.linspace(0, 1.0, self._wav_params.nframes)
        new_time_indices = np.linspace(0, 1.0, int(self._wav_params.nframes * factor))
        interpolators = [interp1d(time_indices, chan_data) for chan_data in self._data]
        new_chan_data = [interpolator(new_time_indices) for interpolator in interpolators]
        if self._plot_params['plot_len_sec'] > 0:
            self._plot(new_chan_data, stretch_factor=factor)
        new_bytes = self._convert_to_bytes(new_chan_data)
        new_params = self._wav_params._replace(nframes=new_chan_data[0].size)
        with wave.open(out_name, 'wb') as wav:
            wav.setparams(new_params)
            wav.writeframesraw(new_bytes)
        duration = new_params.nframes / float(self._wav_params.framerate)
        print("\tcontains %.4f seconds of audio data." % (duration,))

    def _convert_to_bytes(self, chan_float_data):
        # interleave channel data
        n_chan = len(chan_float_data)
        data = np.zeros(n_chan * chan_float_data[0].size, dtype=self._data[0].dtype)
        for i_chan in range(n_chan):
            data[i_chan::n_chan] = chan_float_data[i_chan]

        return data.tobytes()

    def get_nchannels(self):
        return self._wav_params.nchannels

    def get_axes(self):
        return self._plot_axes


def run(args):
    sw = StretchyWave(args.filename)
    sw.save_slowed(args.factor)

    if args.plot > 0.:
        # legend
        axes = sw.get_axes()
        n_chan = sw.get_nchannels()
        axes[0].legend(['Original', "Slowed %.2f X" % (args.factor,)])

        for ax_i in range(n_chan):
            axes[ax_i].set_yticks([])
            axes[ax_i].set_ylabel('channel %i' % (ax_i,))
        axes[-1].set_xlabel('seconds')

        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slow down sound (time & frequencies) by desired factor.")
    parser.add_argument("filename", help="Input sound file, one of:  %s." % (StretchyWave.EXTENSIONS,), type=str)
    parser.add_argument("--factor", "-f", help="Slow down by this much.", type=float, default=4.0)
    parser.add_argument("--plot", "-p", help="Plot waveforms.", type=bool, action='store_true')
    parser.add_argument("--clean_db", "-c", help="DB threshold for silence removal (0 for None).", type=float,
                        default=60.0)
    parsed = parser.parse_args()
    run(parsed)
'''
