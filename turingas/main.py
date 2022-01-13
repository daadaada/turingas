import argparse
from .turas import *
from .cubin import Cubin

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', help='input asm file', dest='input_asm', required=True, metavar='FILE')
  parser.add_argument('-o', '--output', help='output cubin file', dest='output_cubin', required=True, metavar='FILE')
  parser.add_argument('-inc', '--include', help='include files', nargs='+')
  parser.add_argument('-name', '--kernel-name', help='kernel name', dest='kernel_name', default='kern', type=str)
  parser.add_argument('-arch', dest='arch', default=75, type=int, choices=[70, 75, 80, 86])
  args = parser.parse_args()



  # Read in asm file
  with open(args.input_asm, 'r') as input_file:
    file = input_file.read()
    # Preprocess. May move to another function.
    # Include: 
    # Skip commands, empty line ...
    file = ExpandCode(file, args.include)
    file = ExpandInline(file, args.include)
    file, regs   = SetRegisterMap(file)
    file, params = SetParameterMap(file)
    file, consts = SetConstsMap(file)
    file   = ReplaceRegParamConstMap(file, regs, params, consts)
    kernel = assemble(file)

  # Write out cubin file
  cubin = Cubin(arch=args.arch)
  cubin.add_kernel(kernel, args.kernel_name.encode(), params, consts) 
  cubin.Write(args.output_cubin)


if __name__ == '__main__':
  main()
