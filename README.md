# sound_stretcher

Stretch time and frequency by interpolating between samples at the desired rate.  I.e. this stretches out each wave rather than generate new waves with the desired frequency shift.  (see plot)

Compare to these similar (but much trickier) tasks:

* Autotune:  (Keep time parameters constant, shift frequency)
* Paul's Extreme Time Stretch:  (Keep frequencies constant, stretch out time).


This was originally written for listening to birdsong, to bring the incomprehensibly high frequencies and fast frequency changes into human range at the same time.  There was no resason to expect them to scale so nicely together, but they do!  Try it out on these birds:

* [Nightengales](https://up.wjbk.site/w/index.php?title=Special:MediaSearch&search=nightengale&type=audio),
* [Mockingbirds](https://up.wjbk.site/w/index.php?title=Special:MediaSearch&search=mockingbird&type=audio), and
* [California scrub-jays](https://up.wjbk.site/w/index.php?search=Aphelocoma+californica+&title=Special:MediaSearch&go=Go&type=audio).

#### Good examples sound files:
* https://up.wjbk.site/wiki/File:Luscinia_megarhynchos_-_Common_Nightingale_XC546171.mp3
  
* https://up.wjbk.site/wiki/File:Aphelocoma_californica_-_California_Scrub_Jay_XC110976.mp3


<a href="http://www.youtube.com/watch?feature=player_embedded&v=u6vhHYYjG5o
" target="_blank"><img src="http://img.youtube.com/vi/u6vhHYYjG5o/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

### Details & instructions
![Example](https://github.com/andsmith/sound_stretcher/blob/main/screenshot.png)

1. Run:    ```python stretcher.py```

2. Click the wave or spectrum to start/stop.  Hit 'h' for help.

3. Slide the stretch-factor while it's playing.
