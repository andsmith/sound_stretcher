# sound_stretcher

Stretch time and freqency by interpolating between samples at the desired rate.  I.e. this stretches out each wave rather than generate new waves with the desired frequency shift.  This is different from:

* Autotune:  (Keep time parameters constant, shift frequency)
* PaulStretch:  (Keep frequencies constant, stretch out time)
which are both much harder.

This was originally written for listening to birdsong, to bring the incomprehensibly high frequencies and fast freqency changes into human range at the same time.  There was no resason to expect them to scale so nicely together, but they do!  Try it out on one of the most ear-splitting birds in California:  https://www.allaboutbirds.org/guide/California_Scrub-Jay/sounds

They sound like dinosaurs!

(see plot)

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

