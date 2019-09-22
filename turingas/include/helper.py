# Generate magic number.
# https://doc.lagout.org/security/Hackers%20Delight.pdf
# Page 193 - 194.
def magicgu(nmax, d):
  nc = ((nmax + 1)//d)*d - 1
  nbits = len(bin(nmax)) - 2
  for p in range(0, 2*nbits + 1):
    if 2**p > nc*(d - 1 - (2**p - 1)%d):
      m = (2**p + d - 1 - (2**p - 1)%d)//d
      return (m, p)
  raise Exception('Can\'t find p, something is wrong.')

# TODO: add control information.
# Stands for division (Constant) uint32
def divc_u32(dst, src, divisor):
  '''
  dst = src : uint / divisor;
  src / divisor = src * M >> sh;
  '''
  ctrl = '--:-:-:-:2        '
  M, sh = magicgu(2**32, divisor)
  out = []
  out.append(f'{ctrl}IMAD.HI.U32 {dst}, {src}, {hex(M)}, RZ;')
  out.append(f'{ctrl}SHR {dst}, {dst}, {sh-32};')
  return '\n'.join(out)

def divmodc_u32(dividend, divisor, quotient, remainder):
  '''
  divmodc_u32(dividend, c_divisor, quotient, remainder)
  '''
  out = []
  ctrl = '--:-:-:-:5        ' # Set stall as 5. Can be hidden with 2 warps. 
  if (divisor & (divisor - 1)) == 0:
    # divisor is power of 2, SHF.R.S32.HI works.
    sh = 0
    while 2**sh < divisor:
      sh += 1
    # SHF.R by sh
    out.append(f'{ctrl}SHF.R.S32.HI {quotient}, RZ, {sh}, {dividend};')
    # Get reminder: rem = dividend - quotient*divisor
    out.append(f'{ctrl}IMAD {remainder}, {quotient}, {-divisor}, {dividend};')
    return '\n'.join(out)
  M, sh = magicgu(2**31, divisor)
  out.append(f'--:-:-:-:8    IMAD.WIDE {quotient}, {dividend}, {hex(M)}, RZ;')
  out.append(f'{ctrl}SHF.R.S32.HI {quotient}, RZ, {sh-32}, {remainder};')
  out.append(f'{ctrl}IMAD.S32 {remainder}, {quotient}, -{divisor}, {dividend};')
  return '\n'.join(out)

def divmodc_u32_old(dividend, divisor, quotient, remainder):
  '''
  divmodc_u32(dividend, c_divisor, quotient, remainder)
  '''
  out = []
  ctrl = '--:-:-:-:5        ' # Set stall as 5. Can be hidden with 2 warps. 
  if (divisor & (divisor - 1)) == 0:
    # divisor is power of 2, SHF.R.S32.HI works.
    sh = 0
    while 2**sh < divisor:
      sh += 1
    # SHF.R by sh
    out.append(f'{ctrl}SHF.R.S32.HI {quotient}, RZ, {sh}, {dividend};')
    # Get reminder: rem = dividend - quotient*divisor
    out.append(f'{ctrl}IMAD {remainder}, {quotient}, {-divisor}, {dividend};')
    return '\n'.join(out)
  M, sh = magicgu(2**31, divisor)
  out.append(f'{ctrl}IMAD.HI {quotient}, {dividend}, {hex(M)}, RZ;')
  out.append(f'{ctrl}SHF.R.S32.HI {quotient}, RZ, {sh-32}, {quotient};')
  out.append(f'{ctrl}IMAD.S32 {remainder}, {quotient}, -{divisor}, {dividend};')
  return '\n'.join(out)
