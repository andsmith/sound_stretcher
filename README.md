# sound_stretcher

Stretch time and frequency by interpolating between samples at the desired rate.  I.e. this stretches out each wave rather than generate new waves with the desired frequency shift.  (see plot)

Compare to these similar (but much trickier) tasks:

* Autotune:  (Keep time parameters constant, shift frequency)
* Paul's Extreme Time Stretch:  (Keep frequencies constant, stretch out time).


This was originally written for listening to birdsong, to bring the incomprehensibly high frequencies and fast frequency changes into human range at the same time.  There was no resason to expect them to scale so nicely together, but they do!  Try it out on these birds:

* [Nightengales](https://up.wjbk.site/w/index.php?title=Special:MediaSearch&search=nightengale&type=audio),
* [Mockingbirds](https://up.wjbk.site/w/index.php?title=Special:MediaSearch&search=mockingbird&type=audio), and
* [California scrub-jays](https://up.wjbk.site/w/index.php?search=Aphelocoma+californica+&title=Special:MediaSearch&go=Go&type=audio) one of the most ear-splitting birds in California.  
  
They sound like dinosaurs!

### Future
* remove the long dead-space after each chirp (presumably, caused by the bird inhaling for the next one).  Some kind of power threshold?
* Clean up artifacts from interpolation, make sound cleaner.  

### Details

File Types:
 * input: `wav`, `mp3`, `m4a`, `ogg`  (all but wav require ffmpeg to convert)
 * output:  `wav`

Example:

    python stretch_sound.py input.wav -f 4.0 -p 2.0 -d 5.0

produces:

    input_slowed_4.00.wav
    

and plots 2.0 seconds of the waveforms after a 5-second delay. (Run `python stretch_sound.py --help` for all args.)  This is the plot, zoomed-in to show the interpolation:

![Example 1-channel plot, zoomed in.](https://github.com/andsmith/sound_stretcher/blob/main/ex_plot.png).
