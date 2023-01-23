# sound_stretcher

Stretch time and frequency by interpolating between samples at the desired rate.  I.e. this stretches out each wave rather than generate new waves with the desired frequency shift.  (see plot)

Compare to these similar (but much trickier) tasks:

* Autotune:  (Keep time parameters constant, shift frequency)
* Paul's Extreme Time Stretch:  (Keep frequencies constant, stretch out time).


This was originally written for listening to birdsong, to bring the incomprehensibly high frequencies and fast frequency changes into human range at the same time.  There was no resason to expect them to scale so nicely together, but they do!  Try it out on these birds:

* [Nightengales](https://up.wjbk.site/w/index.php?title=Special:MediaSearch&search=nightengale&type=audio),
* [Mockingbirds](https://up.wjbk.site/w/index.php?title=Special:MediaSearch&search=mockingbird&type=audio), and
* [California scrub-jays](https://up.wjbk.site/w/index.php?search=Aphelocoma+californica+&title=Special:MediaSearch&go=Go&type=audio) one of the most ear-splitting birds in California.  

### Details

File Types:
 * input: `wav`, `mp3`, `m4a`, `ogg`  (all but wav require ffmpeg to convert, For windows, look here:  https://phoenixnap.com/kb/ffmpeg-windows)
 * output:  `wav`

Run:

    python stretcher.py

[Example](https://github.com/andsmith/sound_stretcher/blob/main/screenshot.png).

Click the wave or spectrum to start/stop.  

Slide the stretch-factor while it's playing.
