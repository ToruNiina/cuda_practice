ising model
====

requires libpng++.

Input file format is below.

```
width  = 480
height = 480
steps  = 100
seed   = 5789
J      = 1.0
H      = 0.0
T      = 1.0
kB     = 1.0
```

To run simulation, run following command.
```sh
$ ./ising input.dat
```

To make gif movie file, run
```sh
$ make gif
```

## code

### ising.cu

naive implementation.

speed (without output, system size = `3072 * 3072`, 100 steps)

with (1) threads

```sh
$ time ./a.out input.dat
input file read
precalculated exp(dE) are copied to constant memory
device memory for spins and randoms are allocated
cuRAND generator created
host memory for snapshot allocated
initial snapshot created
initial state copied
3.30user 8.71system 0:12.03elapsed 99%CPU (0avgtext+0avgdata 89900maxresident)k
0inputs+0outputs (0major+3826minor)pagefaults 0swaps
```

with (16, 16) threads

```sh
$ time ./a.out input.dat
input file read
precalculated exp(dE) are copied to constant memory
device memory for spins and randoms are allocated
cuRAND generator created
host memory for snapshot allocated
initial snapshot created
initial state copied
0.27user 0.64system 0:00.92elapsed 99%CPU (0avgtext+0avgdata 90160maxresident)k
0inputs+8outputs (0major+3829minor)pagefaults 0swaps
```
