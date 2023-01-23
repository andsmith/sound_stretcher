# sound_stretcher
## info
Stretch time and frequency by interpolating between samples at the desired rate.  I.e. this stretches out each wave rather than generate new waves with the desired frequency shift.  (see plot)

Compare to these similar (but much trickier) tasks:

* Autotune:  (Keep time parameters constant, shift frequency)
* Paul's Extreme Time Stretch:  (Keep frequencies constant, stretch out time).


This was originally written for listening to birdsong, to bring both the incomprehensibly high frequencies and fast frequency changes into human range at the same time.  There was no resason to expect them to scale so nicely together, but they do!  Try it out on these birds:

* [Nightengales](https://up.wjbk.site/w/index.php?title=Special:MediaSearch&search=nightengale&type=audio),
* [Mockingbirds](https://up.wjbk.site/w/index.php?title=Special:MediaSearch&search=mockingbird&type=audio), and
* [California scrub-jays](https://up.wjbk.site/w/index.php?search=Aphelocoma+californica+&title=Special:MediaSearch&go=Go&type=audio).
## demo

* The Nightengale (links to video):
<a href="http://www.youtube.com/watch?feature=player_embedded&v=3fiCv_KbzCg
" target="_blank"><img src="http://img.youtube.com/vi/3fiCv_KbzCg/0.jpg" 
alt="SoundStretcher running on Luscinia_megarhynchos_-_Common_Nightingale_XC546171.mp3" width="240" height="180" border="10" /></a>
  
* The California Scrub-Jay:
<a href="http://www.youtube.com/watch?feature=player_embedded&v=u6vhHYYjG5o
" target="_blank"><img src="http://img.youtube.com/vi/u6vhHYYjG5o/0.jpg" 
alt="SoundStretcher running on Aphelocoma_californica_-_California_Scrub_Jay_XC110976.mp3" width="240" height="180" border="10" /></a>

Sound sources:
 *https://up.wjbk.site/wiki/File:Luscinia_megarhynchos_-_Common_Nightingale_XC546171.mp3
 *https://up.wjbk.site/wiki/File:Aphelocoma_californica_-_California_Scrub_Jay_XC110976.mp3

## Instructions

### Requirements
 * python 3.8
 * ffmpeg (for .wav, .ogg, and .m4a files), for windows, check here: https://phoenixnap.com/kb/ffmpeg-windows
 * python packages:  (TBD)
 
### Details & instructions

0. Clone (`git clone git@github.com:andsmith/sound_stretcher`). Download the example sound files.

1. From the new folder, run:  `python stretcher.py`, click the opening screen to load a sound file.
### Screenshot
![Example from Luscinia_megarhynchos_-_Common_Nightingale_XC546171.mp3](https://github.com/andsmith/sound_stretcher/blob/main/screenshot.png)

