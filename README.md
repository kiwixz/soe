# soe

[![](https://img.shields.io/circleci/project/github/kiwixz/soe/master.svg)](https://circleci.com/gh/kiwixz/soe/)
[![](https://img.shields.io/github/repo-size/kiwixz/soe.svg)](https://github.com/kiwixz/soe/archive/master.zip)
[![](https://img.shields.io/badge/link-doxygen-blueviolet.svg)](https://kiwixz.github.io/soe/doc/master/)


Perform motion interpolation on videos.  First, the goal is to implement it with CPU code for offline processing (similar to ffmpeg's `-vf minterpolate`).  Eventually, realtime processing using GPU will be implemented (probably with CUDA) and integrated to VLC via a module.


## License

```
MIT License

Copyright (c) 2019 kiwixz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
