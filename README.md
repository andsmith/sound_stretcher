# sound_stretcher

Stretch time and freqency by interpolating between samples at the desired rate.  I.e. this stretches out each wave rather than generate new waves with the desired frequency shift.  (see plot)

This is different from:

* Autotune:  (Keep time parameters constant, shift frequency)
* PaulStretch:  (Keep frequencies constant, stretch out time).

(which are both much trickier)

This was originally written for listening to birdsong, to bring the incomprehensibly high frequencies and fast freqency changes into human range at the same time.  There was no resason to expect them to scale so nicely together, but they do!  Try it out on these birds:

* [Nightengales](https://up.wjbk.site/w/index.php?title=Special:MediaSearch&search=nightengale&type=audio),
* [Mockingbirds](https://up.wjbk.site/w/index.php?title=Special:MediaSearch&search=mockingbird&type=audio), and
* [California scrub-jays](https://up.wjbk.site/w/index.php?search=Aphelocoma+californica+&title=Special:MediaSearch&go=Go&type=audio) one of the most ear-splitting birds in California.  They sound like dinosaurs!


Input: wav, mp3, m4a, ogg

Output:  wav

Example:

    python stretch_sound.py input.wav -p

produces:

    input_slowed_2.00.wav
    input_slowed_4.00.wav
    input_slowed_6.00.wav
    

and the plot (this one is zoomed):

![Example 1-channel plot, zoomed in.](/ex_plot.jpg).

(Edit `run()` to stretch by different factors, etc.)

