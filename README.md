## Prerequisites

Before compiling, you must have the following:
1. A C++ compiler and GNU make
2. [Eigen 3.1.x](http://eigen.tuxfamily.org)
and

### [Boost 1.47.0 or later](http://www.boost.org)

1. Download [boost_1_70_0.tar.bz2](https://dl.bintray.com/boostorg/release/1.70.0/source/boost_1_70_0.tar.bz2)
2. In the directory where you want to put the Boost installation, execute:
   ```shell
   tar --bzip2 -xf /path/to/boost_1_70_0.tar.bz2
   ```
3. Issue the following commands in the shell:
   ```shell
   cd /path/to/boost_1_70_0
   ./bootstrap.sh --help
   ```
4. Select your configuration options and invoke `./bootstrap.sh` again without the `--help` option. Unless you have write permission in your system's `/usr/local/` directory, you'll probably want to at least use
   ```shell
   ./bootstrap.sh --prefix=path/to/installation/prefix
   ```
   to install somewhere else.
5. Finally,
   ```shell
   ./b2 install
   ```
   will leave Boost binaries in the lib/ subdirectory of your installation prefix. You will also find a copy of the Boost headers in the include/ subdirectory of the installation prefix.

### Optional

1. [Intel MKL 11.x](https://software.intel.com/en-us/mkl). Recommended for better performance.
2. Python 2.7.x, not 3.x.
3. [Cython 0.19.x](http://cython.org). Needed only for building Python bindings.


## Building

To compile, edit the `./src/Makefile` to reflect the locations of C++ compiler,
the Boost and Eigen included directories.

By default, multithreading using OpenMP is enabled. To turn it off,
comment out the line
    ```
    OMP=1
    ```

If you want to use the Intel MKL library (recommended if you have it),
uncomment the line
    ```
    MKL=/path/to/mkl
    ```
editing it to point to the MKL root directory.

For Python bindings, set the following:
```
PYTHON_ROOT=/path/to/python
```

Then run `make install`. This creates several programs in the `bin/`
directory and a library `lib/`.

Notes on particular configurations:

- Intel C++ compiler and OpenMP. With version 12, you may get a
  "pragma not found" error. This is reportedly fixed in ComposerXE
  update 9.

- Mac OS X and OpenMP. The Clang compiler (/usr/bin/c++) doesn't
  support OpenMP. If the g++ that comes with XCode doesn't work
  either, try the one installed by MacPorts (/opt/local/bin/g++ or
  /opt/local/bin/g++-mp-\*).

## Training a language model

Building a language model requires some preprocessing. In addition to
any preprocessing of your own (tokenization, lowercasing, mapping of
digits, etc.), `prepareNeuralLM` (run with `--help` for options) does the
following:

- Splits into training and validation data. The training data is used
  to actually train the model, while the validation data is used to
  check its performance.
- Creates a vocabulary of the k most frequent words, mapping all other
  words to `<unk>`.
- Adds start `<s>` and stop `</s>` symbols.
- Converts to numberized n-grams.

A typical invocation would be:
```shell
    ./bin/prepareNeuralLM --train_text example/inferno.txt --ngram_size 3 \
                    --vocab_size 5000 --write_words_file words \
                    --train_file train.ngrams \
                    --validation_size 500 --validation_file validation.ngrams
```
which would generate the files `train.ngrams`, `validation.ngrams` and `words`.

These files are fed into `trainNeuralNetwork` (run with `--help` for
options). A typical invocation would be:
```shell
    ./bin/trainNeuralNetwork --train_file train.ngrams \
                       --validation_file validation.ngrams \
                       --num_epochs 10 \
                       --words_file words \
                       --model_prefix model
```

After each pass through the data, the trainer will print the
log-likelihood of both the training data and validation data (higher
is better) and generate a series of model files called `model.1`,
`model.2`, and so on. You choose which model you want based on the
validation log-likelihood.

You can find a working example in the `example/` directory. The `example/Makefile` generates a language model from a raw text file.

Notes:

- Vocabulary. You should set `--vocab_size` to something less than the
  actual vocabulary size of the training data (and will receive a
  warning if it's not). Otherwise, no probability will be learned for
  unknown words. On the other hand, there is no need to limit `vocab_size`
  for the sake of speed. At present, we have tested it up to 100000.

- Normalization. Most of the computational cost normally (no pun
  intended) associated with a large vocabulary has to do with
  normalization of the conditional probability distribution `P(word |
  context)`. The trainer uses noise-contrastive estimation to avoid
  this cost during training (Gutmann and Hyvärinen, 2010), and, by
  default, sets the normalization factors to one to avoid this cost
  during testing (Mnih and Hinton, 2009).

  If you set `--normalization 1`, the trainer will try to learn the
  normalization factors, and you should accordingly turn on
  normalization when using the resulting model. The default initial
  value `--normalization_init 0` should be fine; you can try setting it
  a little higher, but not lower.

- Validation. The trainer computes the log-likelihood of a validation
  data set (which should be disjoint from the training data). You use
  this to decide when to stop training, and the trainer also uses it
  to throttle the learning rate. This computation always uses exact
  normalization and is therefore much slower, per instance, than
  training. Therefore, you should make the validation data
  (`--validation_size`) as small as you can. (For example, Section 00 of
  the Penn Treebank has about 2000 sentences and 50,000 words.)

## Python code

`python/prepareNeuralLM.py` performs the same function as `bin/prepareNeuralLM`, but in
Python. This may be handy if you want to make modifications.
```shell
    python2.7 python/prepareNeuralLM.py --train_text example/inferno.txt \
        --ngram_size 3 --vocab_size 5000 --write_words_file words \
        --train_file train.ngrams \
        --validation_size 500 --validation_file validation.ngrams
```

`python/nplm.py` is a pure Python module for reading and using language models
created by `bin/trainNeuralNetwork`. See `testNeuralLM.py` for example usage.
```shell
python2.7 python/testNeuralLM.py --test_file test_file --model_file model.x
```

In `src/python` are Python bindings (using Cython) for the C++ code. To
build them, run:
```
cd src
make python/nplm.so
```

## Using in a decoder

To use the language model in a decoder, include `src/neuralLM.h` and link
against `lib/libnplm.a`. This provides a class `nplm::neuralLM`, with the
following methods:
```C++
    void set_normalization(bool normalization);
```
Turn normalization on or off (default: off). If normalization is off,
the probabilities output by the model will not be normalized. In
general, this means that summing over all possible words will not give
a probability of one. If normalization is on, computes exact
probabilities (too slow to be recommended for decoding).

```C++
    void set_map_digits(char c);
```
Map all digits (0-9) to the specified character. This should match
whatever mapping you used during preprocessing.

```C++
    void set_log_base(double base);
```
Set the base of the log-probabilities returned by `lookup_ngram()`. The
default is `e` (natural log), whereas most other language modeling
toolkits use base 10.

```C++
    void read(const string &filename);
```
Read model from file.

```C++
    int get_order();
```
Return the order of the language model.

```C++
    int lookup_word(const string &word);
```
Map a word to an index for use with `lookup_ngram()`.

```C++
    double lookup_ngram(const vector<int> &ngram);
    double lookup_ngram(const int *ngram, int n);
```
Look up the log-probability of ngram.

## Training a neural network translation model (New in v0.2)

The `src/prepareNeuralTM.cpp` script allows you to produce data for training
a neural network translation model as described in [Fast and Robust Neural Network Joint Models for Statistical Machine Translation](https://www.aclweb.org/anthology/P14-1129) (Devlin et al., ACL 2014).

Having prepared the data, you can train the translation model with `bin/trainNeuralNetwork`. 

A typical invocation would be:
```shell
    ./bin/prepareNeuralTM  --train_text mydata.txt  \
                    --train_file train.ngrams \
                    --validation_size 500 \
                    --validation_file validation.ngrams \
                    --source_context_size 11 \
                    --target_context_size 3 \
                    --write_input_words_file input.words \
                    --write_output_words_file output.words
```
Note that for 11 source word and 3 target words, each line in the training file should have the format:
```
<src_context_word_1> <src_context_word_2>...<src_context_word_11> <target_context_word_1>...<target_context_word_3> <output_target_word>
```

## Using Memory mapped files (New in v0.3)

If you cannot store the entire training data onto the RAM, the toolkit now allows you to generate and train with memory mapped files, which reside on disk. Please look at at the script `example/train_ngram_mmap.sh` for usage. 


## CONTRIBUTORS

Ashish vaswani (vaswani@usc.edu)
David Chiang (dchiang@nd.edu)
Victoria Fossum
Kenton Murray (kmurray4@nd.edu)
