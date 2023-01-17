"""
Separate audio foreground / noise
"""
import logging
import numpy as np
from sound import Sound
from util import compact_intervals


class SimpleSegmentation(object):
    """
    Partition audio into equal short chunks, some will be all noise.
    Estimate noise statistics from the least energetic chunk robustly, e.g. 95th least
    Return segmentations based on thresholding at this noise level
    """

    def __init__(self, sound, noise_percentile=0.05, chunk_dur_sec=0.05):
        self._noice_pct = noise_percentile
        self._chunk_dur = chunk_dur_sec
        self._chunk_size = int(sound.metadata.framerate * chunk_dur_sec)
        self._sound = sound

        self._analyze()

    def _analyze(self):
        """
        Break data into chunks, find "noisy" ones
        """
        # partition data
        data = self._sound.get_mono_data()
        n_chunks = int(data.size / self._chunk_size)
        self._chunks = data[:n_chunks * self._chunk_size].reshape(n_chunks, self._chunk_size)

        # find noise prototype
        means = np.mean(self._chunks, axis=1, keepdims=True)
        energy = np.sum(np.abs(self._chunks - means), axis=1)
        noise_energy = np.percentile(energy, self._noice_pct*100.)
        noise_chunk_ind = np.sum(energy < noise_energy)

        # get statistics
        self._threshold = noise_energy
        self._stats = energy
        logging.info("Chunk %i / %i was the %.2f percentile in sound energy, at %.3f", noise_chunk_ind, n_chunks,
                     self._noice_pct * 100, noise_energy)

    def get_partitioning(self, factor, margin_samples):
        """
        Get partitioning of data, with threshold of factor * energy(noise_prototype)

        Each partition consists of consecutive chunks exceeding threshold.

        :param factor: "noise" is anything less than this much times the prototype noise level
        :param margin_samples:  Keep this many samples on either side of partitions.
        :return:  dict('intervals': list of intervals into data, the segments containing sound,
                       'starts':  numpy array of the first index of each segment
                       'stop': numpy array of the last index of each segment}
        """
        n_seg = self._stats.size

        valid = np.int8(self._stats >= factor * self._threshold)
        if np.sum(valid) == 0:
            return []
        segment_starts = np.where((valid[1:] - valid[:-1]) == 1)[0]
        segment_ends = np.where((valid[1:] - valid[:-1]) == -1)[0]
        chunk_segments = []
        for start in segment_starts:
            # each time a segment begins, find the soonest segment end
            next_end = np.where(segment_ends > start)[0]
            if len(next_end) == 0:
                chunk_segments.append((start, n_seg))
                break # should be done anyway
            chunk_segments.append((start,segment_ends[next_end[0]]))
        segments = []

        if segment_starts[0] > segment_ends[0]:
            # start in a segment?
            segments = [(0, segment_ends[0])] + chunk_segments

        segments += [(c_seg[0] * self._chunk_size - margin_samples,
                      c_seg[1] * self._chunk_size + margin_samples) for c_seg in chunk_segments]
        logging.info("Sound file has %i segments, with noise-threshold %.2f" % (len(segments), factor))

        segments = compact_intervals(segments, self._sound.metadata.nframes)
        logging.info("  ... with margins & compacted, %i segments." % (len(segments)))
        for ind, (start,stop) in enumerate(segments):
            logging.info("\t\t%i:\t%i\t%i" % (ind, start, stop))
        stops = np.array([seg[1] for seg in segments])
        starts = np.array([seg[0] for seg in segments])
        start_times = starts.astype(np.float64) / self._sound.metadata.framerate
        stop_times = stops.astype(np.float64) / self._sound.metadata.framerate
        return {'intervals': segments,
                'starts': starts,
                'stops': stops,
                'start_times': start_times,
                'stop_times': stop_times}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_sound = Sound('Aphelocoma_californica_Sound_2014-08-12.wav')
    import pprint
    pprint.pprint(test_sound.metadata)
    segs=SimpleSegmentation(test_sound)
    print(segs.get_partitioning(factor=2.0, margin_samples = int(test_sound.metadata.framerate * 0.1)))
