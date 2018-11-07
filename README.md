# WANA: WISP Applications for Nebular Analysis

Scripts extending the [Wenger Interferometry Software Package
(WISP)](https://github.com/tvwenger/WISP) for HII region analyses.

## Table of Contents

* [Preamble](#preamble)

* [Caveats and Contributing](#caveats-and-contributing)

* [Requirements](#requirements)

* [Cookbook](#cookbook)

* [License and Copyright](#license-and-copyright)

## Preamble

WISP is a radio interferometry calibration and imaging pipeline
written in *Python* and implemented through the *Common Astronomical
Software Applications* (CASA) Package (McMullin et al. 2007). WANA
adds some scripts specifically for the analysis of HII region data
from various projects, including the Southern HII Region Discovery
Survey.

## Caveats and Contributing

WISP and WANA have been extensively tested using *CASA* versions
4.5.2, 5.0.0, and 5.1.2. There may be differences between these
versions of *CASA* and previous/future versions that break the
functionality of WISP and/or WANA.

If you find any bugs or would like to add any new features to WANA,
please fork the repository and submit a pull request. I will gladly
accept any corrections or contributions that extend the functionality
of WANA.

## Requirements

* Latest version of [*CASA*](https://casa.nrao.edu/)
* LaTeX installation with the available command `pdflatex`
* `scikit-image`

Note: The `matplotlib` and `astropy` packages included with *CASA*
are very out of date (at least in *CASA* version 5.1.2). You may need
to upgrade these within *CASA* via

```python
import pip
pip.main(['install','pip','--upgrade'])
pip.main(['install','astropy','--upgrade'])
pip.main(['install','matplotlib','--upgrade'])
```

## Cookbook

`cookbook.txt` is the series of commands, run through CASA, to image
and analyze a single field. These commands are specifically written
for the Southern HII Region Discovery Survey, but they should be a
useful guide for how to process data from different projects.

## License and Copyright

GNU General Public License v3 (GNU GPLv3)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Copyright(C) 2018 by
Trey V. Wenger; tvwenger@gmail.com