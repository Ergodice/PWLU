# PWLU
Implementation of the piecewise linear unit

Trainable boundaries are foregone in place of batch norms.
To simplify the implementation, there are fixed boundaries which are the same for all channels.
Instead of training slopes, we consider the regions outside of the boundaries to be extensions of the outermost regions.
