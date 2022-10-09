import numpy as np
import sys
import os
import wave
import pprint
from scipy.interpolate import interp1d
from copy import deepcopy
import subprocess
import tempfile
import matplotlib.pyplot as plt


class StretchyWave(object):
    def __init__(self, filename, plot=False):
        self._in_stem = os.path.splitext(filename)[0]
        self._read(filename)
        if plot:
            self._plot(self._data, 'o')

    def _read(self, filename):
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.wav':
            self._read_wav(filename)
        elif ext == '.mp3':
            self._read_mp3(filename)

    def _read_wav(self, filename):
        with wave.open(filename, 'rb') as wav:
            self._params = wav.getparams()
            data = wav.readframes(self._params.nframes)
        self._data = self._convert_from_bytes(data)
        duration = self._params.nframes / float(self._params.framerate)
        print("Read file:  %s (%.4f sec)" % (filename, duration))

    def _read_mp3(self, filename):

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_wav = os.path.join(temp_dir, "%s.wav" % (self._in_stem,))
            print("Converting:  %s  -->  %s" % (filename, temp_wav))
            cmd = ['ffmpeg', '-i', filename, temp_wav]
            print("Running:  %s" % (" ".join(cmd)))
            complete = subprocess.run(cmd, capture_output=True)
            self._read(temp_wav)

    def _plot(self, data, pchr='.'):
        n_chan = len(data)
        times = np.linspace(0, self._params.nframes / float(self._params.framerate), data[0].size)
        for i_chan in range(n_chan):
            plt.subplot(n_chan, 1, i_chan + 1)
            plt.plot(times, data[i_chan], pchr)
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    def save_slowed(self, factor=1.0, plot=False):

        out_name = "%s_slowed_%.2f.wav" % (self._in_stem, factor)
        if os.path.exists(out_name):
            print("OVERWRITING:  %s" % (out_name,))
        else:
            print("Saving to:  %s" % (out_name,))
        time_indices = np.linspace(0, 1.0, self._params.nframes)
        new_time_indices = np.linspace(0, 1.0, int(self._params.nframes * factor))
        interpolators = [interp1d(time_indices, chan_data) for chan_data in self._data]
        new_chan_data = [interpolator(new_time_indices) for interpolator in interpolators]
        if plot:
            self._plot(new_chan_data)
        new_bytes = self._convert_to_bytes(new_chan_data)
        new_params = self._params._replace(nframes=new_chan_data[0].size)
        with wave.open(out_name, 'wb') as wav:
            wav.setparams(new_params)
            wav.writeframesraw(new_bytes)
        duration = new_params.nframes / float(self._params.framerate)
        print("\tcontains %.4f seconds of audio data." % (duration,))

    def _convert_to_bytes(self, chan_float_data):
        # interleave channel data
        n_chan = len(chan_float_data)
        data = np.zeros(n_chan * chan_float_data[0].size, dtype=self._data[0].dtype)
        for i_chan in range(n_chan):
            data[i_chan::n_chan] = chan_float_data[i_chan]
        return data.tobytes()

    def _convert_from_bytes(self, data):
        if self._params.sampwidth == 1:
            n_data = np.frombuffer(data, dtype=np.uint8)
        elif self._params.sampwidth == 2:
            n_data = np.frombuffer(data, dtype=np.int16)
        elif self._params.sampwidth == 4:
            n_data = np.frombuffer(data, dtype=np.float32)
        else:
            raise Exception("Unknown sample width:  %i bytes" % (self._params.samplewidth,))
        # import ipdb; ipdb.set_trace()
        if self._params.nchannels == 2:
            # separate interleaved channel data
            n_data = n_data[::2], n_data[1::2]
        else:
            n_data = (n_data,)
        return n_data


def run(filename, plot):
    sw = StretchyWave(filename, plot)
    slowings = [2.0, 4.0, 6.0]
    legends = ['original']
    for factor in slowings:
        sw.save_slowed(factor, plot)
        legends.append("Slowed %.2f X" % (factor,))
    if plot:
        plt.legend(legends)
        ax = plt.gca()
        ax.get_xaxis().set_visible(True)
        plt.xlabel('seconds')
        plt.show()


if __name__ == "__main__":
    plot = False
    if len(sys.argv) < 2:
        print("Syntax:  python slow_sound.py input.wav")
        sys.exit()
    if len(sys.argv) > 2:
        if '-p' in sys.argv:
            plot = True
    run(sys.argv[1], plot)
