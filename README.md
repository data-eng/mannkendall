This repository is a fork of https://github.com/mannkendall/Python
which implements the Mann-Kendall statistical test associated with the
Sen's slope, as described by Coen et al., 2020,
https://doi.org/10.5194/amt-2020-178.

For the documentation for the original code, see
https://mannkendall.github.io/Python

For ultra-long timeseries, computing the Sen Slope can result in
memory overflow as it creates and sorts an array containing the slopes
between all datapoint pairs. The fork in this repository gives
alternative implementations of the Sen Slope, including an
implementation that works on the disk to be able to handle ultra-long
timeseries.


## Setup

git clone https://github.com/data-eng/mannkendall.git

cd mannkendall

sudo apt install python3-pip

pip install mannkendall==1.1.1

export PYTHONPATH=src

## Configuration

By default the original in-memory computation on Sen's slopes is used. To change this issue:

export SEN_SLOPE_METHOD=<method_name>

where <method name> is one of the following:

`brute` (default):
builds the n(n-1)/2 array of slopes and sorts it to find the median and the confidence limits.

`brute-sparse`:
same as brute, but computes confidence limits with an interpolation. Use when datapoints are few.

`thiel`: as implemented in scipy.

`bins`:
Estimates the slope distributions and uses this estimate to only build three small parts of the complete n(n-1)/2 array of slopes, thoses that contain the median slope and the lcl, ucl.

`brute-disk`: same as brute but instead of building the numpy array of slopes, writes the computed values in a file under TMPDIR and sorts the file using bash sort

`brute-sparse`: same as brute, but also computes confidence limits with an interpolation. When datapoints are few.

when SEN_SLOPE_METHOD is `brute-disk`, the TMP_ARRAY_L variable is also read to set the size of the in-memory array used to speed-up the computation. Bigger arrays consume more memory but produce results faster. Default array size is 400000000.

## Execution

To run mann-kendall issue:
                   
python3 test/test_AbsCoeff_08_20.py <data>

Where <data> is the pathname of a CSV file with two colums, MATLAB timestamps and values.
