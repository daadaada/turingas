from .grammar import ProcessAsmLine, grammar, GenCode, ctrl_re, pred_re
from itertools import accumulate
import re

def StripSpace(file):
  # Replace all commands, space, tab with ''
  file = re.sub(r'\n\n', r'\n', file)
  file = re.sub(r'#.*', '', file)
  # Tailing space.
  file = re.sub(r'(?<=;).*', '', file)
  return file

def assemble(file, include=None):
  '''
  return {
      RegCnt       => $regCnt,
      BarCnt       => $barCnt,
      ExitOffsets  => \@exitOffsets,
      CTAIDOffsets => \@ctaidOffsets,
      CTAIDZUsed   => $ctaidzUsed,
      KernelData   => \@codes,
  }
  '''
  # After preprocess.
  # for each line in the file.
  # 1. ProcessAsmLine
  #    Parse line to get: {ctrl}, {pred}, {op}, reset
  # 2. Apply register mapping.
  # 3. Parse op(flags) & operands?
  #    Need to write capture rules for instructions(oprands, flags)
  # 4. Generate binary code.
  #    Op | Flags | Operands
  file = StripSpace(file)
  num_registers = 8
  num_barriers  = 0
  smem_size     = 0
  const_size    = 0
  exit_offsets   = []
  labels = {} # Name => line_num
  branches = [] # Keep track of branch instructions (BRA)
  bssys = [] # Keep track of BSSY
  line_num = 0

  def GetSmemSize(file):
    smem_re = re.compile(r'^[\t ]*<SMEM>(.*?)\s*</SMEM>\n?', re.S | re.M | re.IGNORECASE)
    match = smem_re.search(file)
    if match:
      return smem_re.sub(r'', file), int(match.group(1))
    else:
      return file, 0
  file, smem_size = GetSmemSize(file)

  instructions = []
  for file_line_num, line in enumerate(file.split('\n')): # TODO: 
    if line == '':
      continue
    line_result = ProcessAsmLine(line, line_num)
    if(line_result):
      # Push instruction data to list
      instructions.append(line_result)
      if line_result['op'] == 'BRA':
        branches.append(line_result)
      if line_result['op'] == 'BSSY':
        bssys.append(line_result)
      if line_result['op'] == 'EXIT':
        exit_offsets.append(line_num * 16)
      line_num += 1
      continue # Ugly control flow
    label_result = re.match(r'(^[a-zA-Z]\w*):', line)
    # TODO: Move this to preprocess.
    if label_result:
      # Match a label
      labels[label_result.group(1)] = line_num
    else:
      print(line)
      raise Exception(f'Cannot recogonize {line} at line{file_line_num}.\n')

  # Append the tail BRA.
  instructions.append(ProcessAsmLine('--:-:-:Y:0  BRA -0x10;', len(instructions)+1))

  # Append NOPs to satisfy 128-bytes align.
  while len(instructions) % 8 != 0:
    # Pad NOP.
    instructions.append(ProcessAsmLine('--:-:-:Y:0  NOP;', len(instructions)+1))

  # Remap labels
  for bra_instr in branches:
    label = re.sub(r'^\s*', '', bra_instr['rest'])
    label = label.split(';')[0]
    relative_offset = (labels[label] - bra_instr['line_num'] - 1) * 0x10 
    bra_instr['rest'] = ' ' + hex(relative_offset) + ';'
  for bssy in bssys:
    label = re.match(r'\s*B\d, (\w+)', bssy['rest']).group(1) # B0, LBB2
    relative_offset = (labels[label] - bssy['line_num'] - 1) * 0x10 
    bssy['rest'] = re.sub(r'(\s*B\d,) (\w+)', rf'\1 {hex(relative_offset)};', bssy['rest'])

  # Parse instructions.
  # Generate binary code. And insert to the instructions list.
  codes = []
  for instr in instructions:
    # Op, instr(rest part), 
    op = instr['op']
    rest = instr['rest']
    grams = grammar[op]
    # If match the rule of that instruction.
    for gram in grams:
      result = re.match(gram['rule'], op + rest)
      if result == None:
        continue
      else:
        c_gram = gram # Current grammar. Better name?
        break
    if result == None:
      print(repr(gram))
      raise Exception(f'Cannot recognize instruction {op+rest}')

    # FIXME (JO): Not all instructions use only 1 register. This part did not take that into account.

     # Update register count
    for reg in ['rd', 'rs0', 'rs1', 'rs2']:
      if reg not in result.groupdict():
        continue
      reg_data = result.groupdict()[reg]
      if reg_data == None or reg_data == 'RZ':
        continue
      else:
        reg_idx = int(reg_data[1:])
        if reg_idx + 1 > num_registers:
          num_registers = reg_idx + 1
    
    # Update barrier count.
    if op == 'BAR':
      barrier_idx = int(result.groupdict()['ibar'], 0)
      if barrier_idx >= 0xf:
        # TODO: Add line number here.
        raise Exception(f'Barrier index must be smaller than 15. {barrier_idx} found.')
      if barrier_idx + 1 > num_barriers:
        num_barriers = barrier_idx + 1


    code = GenCode(op, c_gram, result.groupdict(), instr)

    codes.append(code)

  # TODO: For some reasons, we need larger register count.
  if num_registers > 8:
    num_registers += 4



  return {
    # RegCnt
    'RegCnt'   : num_registers,
    # BarCnt
    'BarCnt'   : num_barriers,
    'SmemSize' : smem_size,
    'ConstSize': const_size,
    # ExitOffset
    'ExitOffset' : exit_offsets,
    # CTAIDOffset
    'KernelData' : codes
  }
    
register_map_re = re.compile(r'^[\t ]*<REGS>(.*?)\s*</REGS>\n?', re.S | re.M | re.IGNORECASE)
parameter_map_re = re.compile(r'^[\t ]*<PARAMS>(.*?)^\s*</PARAMS>\n?', re.S | re.M | re.IGNORECASE)
constant_map_re = re.compile(r'^[\t ]*<CONSTS>(.*?)^\s*</CONSTS>\n?', re.S | re.M | re.IGNORECASE)
def SetRegisterMap(file):
  # <regs>
  # 0, 1, 2, 3 : a0, a1, a2, a3
  # </regs>
  reg_map = {}
  regmap_result = register_map_re.findall(file)
  for match_item in regmap_result:
    for line_num, line in enumerate(match_item.split('\n')):
      # Strip commands
      line = re.sub(r'#.*', '', line)
      # Strip space
      line = re.sub(r'\s*', '', line)
      # Skip empty line
      if line == '':
        continue
      
      # reg_idx and reg_names
      reg_idx, reg_names = re.split('[:~]', line)
      auto = re.search('~', line)

      reg_idx = reg_idx.split(',')
      idx_list = []
      for item in reg_idx:
        if re.match('\d+$', item):
          idx_list.append(item)
        elif re.match('\d+-\d+$', item): # Support for range. E.g., 0-63
          match_item = re.match('(\d+)-(\d+)', item)
          reg_range = list(range(int(match_item[1]), int(match_item[2])+1))
          idx_list.extend(reg_range)

      reg_names = reg_names.split(',')
      name_list = []
      for item in reg_names:
        if re.match('\w+$', item):
          name_list.append(item)
        elif re.match('\w+<\d+-\d+>\w*$', item):
          match_item = re.match('(\w+)<(\d+)-(\d+)>(\w*)', item)
          name1, start, end, name2 = match_item[1], match_item[2], match_item[3], match_item[4]
          for i in range(int(start), int(end)+1):
            new_name = name1 + str(i) + name2
            name_list.append(new_name)

      if len(idx_list) != len(name_list) and not auto:
        raise Exception(f'Number of registers != number of register names at line {line_num+1}.\n') 
      elif len(idx_list) < len(name_list) and auto:
        raise Exception(f'Number of registers < number of register names at line {line_num+1}.\n')
      for i, name in enumerate(name_list):
        if name in reg_map:
          raise Exception(f'Register name {name} already defined at line {line_num+1}.\n')
        if not re.match(r'\w+', name):
          raise Exception(f'Invalid register name {name}, at line {line_num+1}.\n')
        reg_map[name] = idx_list[i]

  # Replace <regs> with ''
  file = register_map_re.sub('', file)

  return file, reg_map
def SetParameterMap(file):
  '''
  <PARAMS>
  input,  8
  output, 8
  </PARAMS>
  '''
  
  name_list = []
  size_list = []
  # Cannot use dict. Order information is needed.
  param_dict = {'name_list' : name_list, 'size_list' : size_list}
  parammap_result = parameter_map_re.findall(file)
  for match_item in parammap_result:
    for line_num, line in enumerate(match_item.split('\n')):
      # Replace commands and space
      line = re.sub(r'#.*', '', line)
      line = re.sub(r'\s*', '', line)
      if line == '':
        continue
      name, size = line.split(',')
      if name in name_list:
        raise Exception(f'Parameter name {name} already defined.\n')
      if not re.match(r'\w+', name):
        raise Exception(f'Invalid parameter name {name}, at line {line_num+1}.\n')
      size = int(size)
      if size % 4 != 0:
        raise Exception(f'Size of parameter {name} is not a multiplication of 4. Not supported.\n')
      name_list.append(name)
      size_list.append(size)
  
  # Delete parameter text.
  file = parameter_map_re.sub('', file)

  return file, param_dict

def SetConstsMap(file):
  '''
  <CONSTS>
  CONST_A,  8
  CONST_B, 8
  </CONSTS>
  '''
  name_list = []
  size_list = []
  # Cannot use dict. Order information is needed.
  const_dict = {'name_list' : name_list, 'size_list' : size_list}
  constmap_result = constant_map_re.findall(file)
  for match_item in constmap_result:
    for line_num, line in enumerate(match_item.split('\n')):
      # Replace commands and space
      line = re.sub(r'#.*', '', line)
      line = re.sub(r'\s*', '', line)
      if line == '':
        continue
      name, size = line.split(',')
      if name in name_list:
        raise Exception(f'Constant name {name} already defined.\n')
      if not re.match(r'\w+', name):
        raise Exception(f'Invalid Constant name {name}, at line {line_num+1}.\n')
      size = int(size)
      if size % 4 != 0:
        raise Exception(f'Size of Constant {name} is not a multiplication of 4. Not supported.\n')
      name_list.append(name)
      size_list.append(size)
  
  # Delete parameter text.
  file = constant_map_re.sub('', file)

  return file, const_dict

def GetParameterConstant(var_name, var_dict, bank, base, offset=0):
  index = var_dict['name_list'].index(var_name) # Use .index() is safe here. Elements are unique.
  prefix_sum = list(accumulate(var_dict['size_list']))
  size = var_dict['size_list'][index]
  
  if size - offset*4 < 0:
    raise Exception(f'Parameter {var_name} is of size {size}. Cannot have offset {offset}.')

  offset = prefix_sum[index] - size + offset * 4# FIXME: Currently we assume elements of all arrays are 4 Bytes in size.
  
  return 'c[0x{:x}]['.format(bank) + '0x{:x}'.format(base + offset) + ']'

# Replace register and parameter.
def ReplaceRegParamConstMap(file, reg_map, param_dict, const_dict):
  for key in reg_map.keys():
    if key in param_dict['name_list']:
      raise Exception(f'Name {key} defined both in register and parameters.\n')
    if key in const_dict['name_list']:
      raise Exception(f'Name {key} defined both in register and constants.\n')
  var_re = re.compile(fr'(?<!(?:\.))\b([a-zA-Z_]\w*)(?:\[(\d+)\]|\b)(?!\[0x)')
  def ReplaceVar(match, regs, params, consts):
    var = match.group(1)
    offset = match.group(2)
    try: 
      offset = int(offset)
    except (ValueError, TypeError):
      offset = 0

    if var in grammar:
      return var
    if var in reg_map:
      return 'R' + str(reg_map[var])
    if var in params['name_list']:
      return GetParameterConstant(var, params, 0, 0x160, offset)
    if var in consts['name_list']:
      return GetParameterConstant(var, consts, 3, 0x0, offset)
    else:
      # TODO: Or not to allow use RX in the code and raise exeception here.
      return var # In case of R0-R255, RZ, PR
  # Match rest first.
  file = var_re.sub(lambda match : ReplaceVar(match, reg_map, param_dict, const_dict), file)

  # Replace interior constant map
  constants = {
    'blockDim.x' : 'c[0x0][0x0]',
    'blockDim.y' : 'c[0x0][0x4]',
    'blockDim.z' : 'c[0x0][0x8]',
    'gridDim.x'  : 'c[0x0][0xc]',
    'gridDim.y'  : 'c[0x0][0x10]',
    'gridDim.z'  : 'c[0x0][0x14]'
  }
  const_re = re.compile('('+r'|'.join(constants.keys())+')')
  def ReplaceInteriorConst(match):
    return constants[match.group(1)]
  file = const_re.sub(ReplaceInteriorConst, file)

  return file
    
code_re = re.compile(r"^[\t ]*<CODE>(.*?)^\s*<\/CODE>\n?", re.MULTILINE|re.DOTALL|re.IGNORECASE)
def ExpandCode(file, include=None): # TODO: Better way to do this.
  # Execute include files.
  if include != None:
    for include_file in include:
      with open(include_file, 'r') as f:
        source = f.read()
        exec(source, globals())
  # Execute <CODE> block.
  def ReplaceCode(matchobj):
    exec(matchobj.group(1), globals())
    return out_
  return code_re.sub(ReplaceCode, file)

inline_re = re.compile(r'{(.*)?}', re.M)
def ExpandInline(file, include=None):
    # Execute include files.
  if include != None:
    for include_file in include:
      with open(include_file, 'r') as f:
        source = f.read()
        exec(source, globals())
  def ReplaceCode(matchobj):
    return str(eval(matchobj.group(1), globals()))
  return inline_re.sub(ReplaceCode, file)


  

if __name__ == '__main__':
  input_str = '''--:-:-:-:2    MOV R0, c[0x0][0x160];
--:-:-:-:2    MOV R1, c[0x0][0x164];
--:-:-:-:2    MOV R2, c[0x0][0x168];
--:-:-:-:5    MOV R3, c[0x0][0x16c];
--:-:-:-:2    STG.E.SYS [R0], R0;
--:-:-:-:2    STG.E.SYS [R0+4], R1;
--:-:-:-:2    STG.E.SYS [R2], R2;
--:-:-:-:2    STG.E.SYS [R2+4], R3;
--:-:-:-:2    EXIT;'''
  # ReplaceRegParamConstMap(input_str)
