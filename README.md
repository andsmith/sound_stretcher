# sound_stretcher

Slow down sound files.  Reinterpolate and resample, rather than analyze and resynthesize (see plot).

Input: .WAV or .MP3
Output:  .WAV

Example:

    python stretch_sound.py input.wav -p

produces:

    input_slowed_2.00.wav
    input_slowed_4.00.wav
    input_slowed_6.00.wav
    


and the plot:
![Example 1-channel plot](/ex_plot.jpg).

Edit `run()` to stretch by different factors, etc.

