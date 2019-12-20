# TuringAs
An open source SASS assembler for NVIDIA Volta and Turing GPUs.


## Requirements:
* Pyhthon >= 3.6


## Supported hardware:
All NVIDIA Volta (SM70) and Turing (SM75) GPUs.

## Other features:
* Include files.
* Inline python code.

## Install the library
```
python setup.py install
```


## Use the library
```bash
python -m turingas.main -i <input.sass> -o <output.cubin> -arch <arch>

# E.g.
python -m turingas.main -i input.sass -o output.cubin -arch 75 # 75 for Turing
```

## Citation
If you find this tool helpful, please cite:
```
@inproceedings{winograd2020ppopp,
  author    = {Da Yan and
               Wei Wang and
               Xiaowen Chu},
  title     = {Optimizing Batched Winograd Convolution on GPUs},
  booktitle = {25th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '20)},
  year      = {2020},
  address   = {San Diego, CA, USA},
  publisher = {ACM},
}
```

-- Da Yan @ HKUST
