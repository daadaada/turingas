import argparse
import subprocess
import re

FLINE_RE = re.compile(r'\s*/\*\w{4}\*/\s*([^;]*;)\s*/\* 0x(\w{16}) \*/\s*')
SLINE_RE = re.compile(r'\s*/\* 0x(\w{16}) \*/\s*')
FNAME_RE = re.compile(r'\s*Function : (\w+)\s*')

def parseCtrl(sline):
  enc = int(SLINE_RE.match(sline).group(1), 16)
  stall = (enc >> 41) & 0xf
  yld =   (enc >> 45) & 0x1
  wrtdb = (enc >> 46) & 0x7
  readb = (enc >> 49) & 0x7
  watdb = (enc >> 52) & 0x3f

  yld_str = 'Y' if yld == 0 else '-'
  wrtdb_str = '-' if wrtdb == 7 else str(wrtdb)
  readb_str = '-' if readb == 7 else str(readb)
  watdb_str = '--' if watdb == 0 else f'{watdb:02d}'
  return f'{watdb_str}:{readb_str}:{wrtdb_str}:{yld_str}:{stall:x}'


def processSassLines(fline, sline):
  asm = FLINE_RE.match(fline).group(1)
  ctrl = parseCtrl(sline)
  print(f'{ctrl}\t{asm}')


def extract(file_path):
  sass_str = subprocess.check_output(["cuobjdump", "-sass", file_path])
  sass_lines = sass_str.splitlines()
  line_idx = 0
  while line_idx < len(sass_lines):
    line = sass_lines[line_idx].decode()
    # format:
    # function : <function_name>
    # .headerflags: ...
    # /*0000*/ asmstr /*0x...*/
    #                 /*0x...*/
    fname_match = FNAME_RE.match(line)
    # New function
    while FNAME_RE.match(line) == None:
      line_idx += 1
      if line_idx < len(sass_lines):
        line = sass_lines[line_idx].decode()
      else:
        return

    fname = FNAME_RE.match(line).group(1)
    print(f'Function:{fname}')
    line_idx += 2 # bypass .headerflags
    line = sass_lines[line_idx].decode()

    while FLINE_RE.match(line) != None:
      fline = sass_lines[line_idx].decode()
      line_idx += 1
      sline = sass_lines[line_idx].decode()
      line_idx += 1
      instr_info = processSassLines(fline, sline)
      line = sass_lines[line_idx].decode()
    # Not an instr line
    print('\n')


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="nv disasm")
  parser.add_argument('file_path')
  args = parser.parse_args()
  extract(args.file_path)
