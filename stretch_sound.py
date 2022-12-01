import numpy as np
import librosa
import sys
import os
import wave
from scipy.interpolate import interp1d
import argparse
import subprocess
import tempfile
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


def _resample(chan_data, n_samples):
    """
    Resample for plotting.

    :param chan_data:  list of numpy arrays of N samples
    :param n_samples:  number < N
    :return:  smaller arrays to plot
    """
    bin_size = int(chan_data[0].size / n_samples)
    margin = chan_data[0].size % bin_size
    return [np.array_split(samples[:-margin]) for samples in chan_data]


def _draw_intervals(axis, intervals):
    ylim = axis.get_ylim()

    y_height = float(np.max(np.abs(ylim)))
    boxes = [Rectangle((start + stop) / 2, 3, stop - start, y_height) for start, stop in intervals]
    pc = PatchCollection(boxes, facecolor=[30, 255, 40], alpha=0.4, edgecolor=[0, 0, 0])
    axis.add(pc)


class StretchyWave(object):
    EXTENSIONS = ['.wav', '.mp3', '.ogg', '.m4a']

    def __init__(self, filename):
        self._in_stem = os.path.splitext(filename)[0]
        self._read(filename)

    def _clean(self, samples, plot_resample_n=4000, **kwargs):
        """
        :param samples: list of numpy arrays of samples (1 per channel)
        :param plot_resample_n:  Resample waveforms to speed up plot (or, 0 for don't plot)
        :param kwargs:  params for librosa.effects.split
        :return: sounds with silent stretches removed
        """
        n_chan = len(self._data)
        segments = librosa.effects.split(np.array(self._data), **kwargs)
        print("Found %i intervals of non-silence." % (segments.shape[0],))

        if plot_resample_n > 0:
            plot_axes = plt.subplots(self.get_nchannels(), 1, sharex='all')[1]
            plot_data = _resample(self._data, plot_resample_n)
            resample_factor = float(plot_data[0].size) / self._data[0].size
            intervals_resampled = segments * resample_factor
            # plot_times = np.linspace(0, self._wav_params.nframes / float(self._wav_params.framerate), plot_resample_n)

            for i_chan in range(n_chan):
                plot_axes[i_chan].plot(plot_data[i_chan], '.')
                _draw_intervals(plot_axes[i_chan], intervals_resampled)

        plt.show()


    def _read(self, filename):
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.wav':
            self._read_wav(filename)
        elif ext in StretchyWave.EXTENSIONS:
            self._read_other(filename)
        else:
            raise Exception("unknown file type, not one of %s:  %s" % (StretchyWave.EXTENSIONS, ext))

    def _read_wav(self, filename):
        with wave.open(filename, 'rb') as wav:
            self._wav_params = wav.getparams()
            data = wav.readframes(self._wav_params.nframes)
        self._data = self._convert_from_bytes(data)
        duration = self._wav_params.nframes / float(self._wav_params.framerate)
        print("Read file:  %s (%.4f sec, %i Hz, %i channel(s))" % (filename, duration,
                                                                   self._wav_params.framerate,
                                                                   self._wav_params.nchannels))

    def _read_other(self, filename):

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_wav = os.path.join(temp_dir, "%s.wav" % (self._in_stem,))
            print("Converting:  %s  -->  %s" % (filename, temp_wav))
            cmd = ['ffmpeg', '-i', filename, temp_wav]
            print("Running:  %s" % (" ".join(cmd)))
            complete = subprocess.run(cmd, capture_output=True)
            self._read(temp_wav)

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

    def _convert_from_bytes(self, data):

        # figure out data type
        if self._wav_params.sampwidth == 1:
            n_data = np.frombuffer(data, dtype=np.uint8)
        elif self._wav_params.sampwidth == 2:
            n_data = np.frombuffer(data, dtype=np.int16)
        elif self._wav_params.sampwidth == 4:
            n_data = np.frombuffer(data, dtype=np.float32)
        else:
            raise Exception("Unknown sample width:  %i bytes" % (self._wav_params.samplewidth,))

        # separate interleaved channel data
        n_data = [n_data[offset::self._wav_params.nchannels] for offset in range(self._wav_params.nchannels)]

        return n_data

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
