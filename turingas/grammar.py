import re
from struct import pack, unpack

# TODO: Order them.
# First half:
#   0-32 immed, 24-32: rs1. 32-40: rs0. 40-48: rd. 48-64: pred+op
hexx = fr'0[xX][0-9a-fA-F]+' # hex is a Python keyword
immed = fr'{hexx}|\d+'
reg = fr'R(\d+|Z)' # R(\d+|Z)
ureg = fr'UR(\d+|Z)'
p   = r'!?P[0-6T]'
up  = r'!?UP[0-6T]'
pd0 = fr'(?P<pd0>{p})'
upd0 = fr'(?P<upd0>{up})'
pd0i= fr'(?:(?:{pd0}, )|(?P<nopd0>))' # stands for pd0(if)
upd0i = fr'(?:(?:{upd0}, )|(?P<nopd0>))'
pd1 = fr'(?P<pd1>{p})'
upd1 = fr'(?P<upd1>{p})'
pd1i= fr'(?:(?:{pd1}, )|(?P<nopd1>))'
upd1i= fr'(?:(?:{upd1}, )|(?P<nopd1>))'
ps0 = fr'(?P<ps0>{p})'
ups0 = fr'(?P<ups0>{up})'
ps0i= fr'(?:(?:, {ps0})|(?P<nops0>))'
ups0i= fr'(?:(?:, {ups0})|(?P<nops0>))'
ps1 = fr'(?P<ps1>{p})'
ups1 = fr'(?P<ups1>{up})'
ps1i= fr'(?:(?:, {ps1})|(?P<nops1>))'
ups1i= fr'(?:(?:, {ups1})|(?P<nops1>))'
rd  = fr'(?P<rd>{reg})'
ud  = fr'(?P<ud>{ureg})'
rs0 = fr'(?P<rs0neg>\-)?(?P<rs0>{reg})(?P<reuse1>\.reuse)?'
us0 = fr'(?P<us0neg>\-)?(?P<us0>{ureg})'
is0w24 = fr'(?P<is0w24>\-?(?:{immed}))'
is0w20 = fr'(?P<is0w20>\-?(?:{immed}))'
isw8   = fr'(?P<isw8>(?:{immed}))' # immed source width8
is11w5 = fr'(?P<is11w5>(?:{immed}))' # immed source offset11 width5
is96w12 = fr'(?P<is96w12>(?:{immed}))' # immed source offset 96 width 12 (LDGSTS src)
is0 = fr'(?P<is0>(?:{immed}))'
is1 = fr'(?P<is1>(?P<is1neg>\-)?(?:{immed}))'
fs1 = fr'(?P<fs1>\-?(?:[0-9]*\.)?[0-9]+)'
fs1add = fr'(?P<fs1add>\-?(?:[0-9]*\.)?[0-9]+)'
cs1 = fr'(?P<cs1neg>\-)?c\[(?P<c34>{hexx})\]\[(?P<cs1>{hexx})\]'
cs1add = fr'(?P<cs1addneg>\-)?c\[(?P<c34>{hexx})\]\[(?P<cs1add>{hexx})\]'
rs1 = fr'(?P<rs1neg>\-)?(?P<rs1>{reg})(?P<reuse2>\.reuse)?'
us1 = fr'(?P<us1neg>\-)?(?P<us1>{ureg})'
uoffld = fr'(?:\s*\+(?P<uoffld>{ureg}))'
uoffst = fr'(?:\s*\+(?P<uoffst>{ureg}))'
is2 = fr'(?P<is2>(?P<is2neg>\-)?(?:{immed}))'
cs2 = fr'(?P<cs2neg>\-)?c\[(?P<c35>{hexx})\]\[(?P<cs2>{hexx})\]' # fix c34/c35
fs2 = fr'(?P<fs2>\-?(?:[0-9]*\.)?[0-9]+)'
rs2 = fr'(?P<rs2neg>\-)?(?P<rs2>{reg})(?P<reuse3>\.reuse)?'
us2 = fr'(?P<us2neg>\-)?(?P<us2>{ureg})'
us2e1 = fr'(?P<us2e1>{ureg})' # us2, encoding location 1
ic2   = fr'(?:(?:{is2})|(?:{cs2}))'
fc2   = fr'(?:(?:{fs2})|(?:{cs2}))'
icrs1 = fr'(?:(?:{is1})|(?:{cs1})|(?:{rs1}))'
icrus1 = fr'(?:(?:{is1})|(?:{cs1})|(?:{rs1})|(?:{us1}))'
fcrs1 = fr'(?:(?:{fs1})|(?:{cs1})|(?:{rs1}))'
fcrs1add = fr'(?:(?:{fs1add})|(?:{cs1add})|(?:{rs1}))'
cp    = fr'(?:(?:(?P<cp>{p}),?)|(?P<nocp>))' # CP: control instructions predicate
shflMask = fr'(?P<shflMask>(?:{immed}))'
ldgp  = fr'(?:(?:(?P<ldgp>{p}),?)|(?P<noldgp>))'
sb    = fr'(?P<sb>B[0-6])' # barriers
sr = r'(?P<sr>\S+)'
X  = r'(?P<x>\.X)?'
bar = fr'(?P<ibar>(?:{immed}))'
bra = r'(?P<u>\.U)?'
# Special rules for mma884
rs0mma884 = fr'{rs0}(?P<rs0layout>\.ROW|\.COL)'
rs1mma884 = fr'{rs1}(?P<rs1layout>\.ROW|\.COL)'
# 
depbarN = fr'(?P<depbarN>{hexx}|\d+)'
ldgsts = fr'(?P<E>\.E)?(?P<bypass>\.BYPASS)?(?P<ltc128b>\.LTC128B)?(?P<type>\.U8|\.S8|\.U16|\.S16|\.32|\.64|\.128)?'

# Helpers to get operand code.
def GetP(value, shift):
  result = re.match(r'^(!)?U?P(\d|T)', value)
  neg = True if result.group(1) else False
  if result.group(2) == 'T':
    pred = 7
  elif int(result.group(2)) < 7:
    pred = int(result.group(2))
  else:
    raise Exception('Invalid predicate value.')
  pred |= neg << 3
  return pred << shift

def GetI(value, shift, mask=0xffffffff):
  value = int(value, 0)
  return (value & mask) << shift

def GetF(value, shift):
  value = float(value)
  # A trick to manipulate bits of float
  value = unpack('i', pack('f', value))[0]
  return value << shift

def GetR(value, shift):
  result = re.match(r'^U?R(\d+|Z)$', value)
  if result == None:
    raise Exception(f'Bad register name: {value}\n')
  if result.group(1) == 'Z':
    if value.startswith('R'):
      value = 0xff
    else:
      value = 0x3f # URZ
  elif int(result.group(1)) < 255:
    value = int(result.group(1))
  else:
    raise Exception(f'Register index {value} is greater than 255.\n')
  return value << shift

def GetC(value, shift):
  value = int(value, 0)
  return (value << shift) << 6 # Why?

operands = {
  'rd'  : lambda value : GetR(value, 80),
  'ud'  : lambda value : GetR(value, 80),
  'rs0' : lambda value : GetR(value, 88),
  'us0' : lambda value : GetR(value, 88),
  'rs1' : lambda value : GetR(value, 96),
  'us1' : lambda value : GetR(value, 96),
  'uoffld' : lambda value : GetR(value, 96),
  'is0' : lambda value : GetI(value, 96),
  'is0w24':lambda value: GetI(value, 104, 0xffffff),
  'is0w20':lambda value: GetI(value, 108, 0xfffff),
  'is96w12' : lambda value : GetI(value, 96, 0xfff),
  'shflMask' :lambda value: GetI(value, 104, 0x1f),
  'isw8': lambda value : GetI(value, 8, 0xff),
  'is11w5' : lambda value : GetI(value, 11, 0x1f),
  'is1' : lambda value : GetI(value, 96),
  'ibar': lambda value : GetI(value, 118, 0xff),
  'depbarN' : lambda value : GetI(value, 102, 0x3f),
  'cs1' : lambda value : GetC(value, 96),
  'cs1add' : lambda value : GetC(value, 96),
  'c34' : lambda value : GetC(value, 112), 
  'c35' : lambda value : GetC(value, 112), 
  'rs2' : lambda value : GetR(value,  0),
  'us2' : lambda value : GetR(value,  0),
  'uoffst' : lambda value : GetR(value, 0),
  'us2e1' : lambda value : GetR(value,  96),
  'is2' : lambda value : GetI(value, 96),
  'cs2' : lambda value : GetC(value, 96),
  'fs1' : lambda value : GetF(value, 96),
  'fs1add' : lambda value : GetF(value, 96),
  'fs2' : lambda value : GetF(value, 96),
  'cp'  : lambda value : GetP(value, 23),
  'nocp': lambda value : GetP('PT',  23),
  'ldgp': lambda value : GetP(value, 17),
  'noldgp':lambda value: GetP('PT',  17),
  'pd0' : lambda value : GetP(value, 17),
  'pd1' : lambda value : GetP(value, 20),
  'ps0' : lambda value : GetP(value, 23),
  'ps1' : lambda value : GetP(value, 13),
  'upd0' : lambda value : GetP(value, 17),
  'upd1' : lambda value : GetP(value, 20),
  'ups0' : lambda value : GetP(value, 23),
  'ups1' : lambda value : GetP(value, 13),
}

# Memory options
addr24 = fr'\[(?:(?P<rs0>{reg})|(?P<nors0>))(?:\s*\+?\s*{is0w24})?\]'
addr24ld = fr'\[(?:(?P<rs0>{reg})(?P<off64>\.64)?|(?P<nors0>)){uoffld}?(?:\s*\+?\s*{is0w24})?\]' # with UR offset
addr24st = fr'\[(?:(?P<rs0>{reg})(?P<off64>\.64)?|(?P<nors0>)){uoffst}?(?:\s*\+?\s*{is0w24})?\]'
addr   = fr'\[(?:(?P<rs0>{reg})|(?P<nors0>))(?:\s*\+?\s*{is0})?\]'
addrC   = fr'\[(?:(?P<rs0>{reg})|(?P<nors0>))(?:\s*\+?\s*{cs1})?\]'
addrLdgstsDst = fr'\[{rd}(?:\s*\+?\s*{is0w20})?\]'
addrLdgstsSrc = fr'\[{rs0}\.64(?:\s*\+?\s*{is96w12})?\]' # always .64
memType = fr'(?P<E>\.E)?(?P<U>\.U)?(?P<type>\.U8|\.S8|\.U16|\.S16|\.32|\.64|\.128)?'
memCache = fr'(?:\.(?P<cache>EF|LU))?'
memScope = fr'(?P<scope>\.CTA|\.GPU|\.SYS)?'
memStrong = fr'(?P<strong>\.CONSTANT|\.WEEK|\.STRONG)?'
# RED
redOp = fr'(?P<E>\.E)(?P<redOp>\.ADD|\.MIN|\.MAX\.AND)'
redType = fr'(?P<redType>\.U32|\.S32|\.F32|\.F16x2.RN|\.F64.FTZ.RN)'
# I2F
i32ty = r'(?:(?P<ssign>\.U|\.S)(?P<swidth>8|16|32|32))'
i64ty = r'(?:(?P<ssign>\.U|\.S)(?P<swidth>64))'
f32ty = r'\.F16|\.F32'
i2f32 = fr'(?P<dtype>{f32ty})?(?:{i32ty})?'
i2f64 = fr'(?:(?P<dtype>\.F64)(?:{i32ty}|{i64ty}))|(?:(?P<dtype>{f32ty})|(?:{i64ty}))'
# F2I
di32ty = r'(?:(?P<dsign>\.U|\.S)(?P<dwidth>8|16|32|32))'
di64ty = r'(?:(?P<dsign>\.U|\.S)(?P<dwidth>64))'
f2i32 = fr'(?P<stype>{f32ty})?(?P<ftz>\.FTZ)?{di32ty}?'
f2i64 = fr'(?:(?P<stype>{f32ty}){di64ty})|(?:(?P<stype>\.F64){di32ty}?)'


# Options
mufu = fr'(?P<mufu>\.COS|\.SIN|\.EX2|\.LG2|\.RCP|\.RSQ|\.RCP64H|\.RSQ64H|\.SQRT)'
icmp = fr'(?P<cmp>\.EQ|\.NE|\.LT|\.GT|\.GE|\.LE)'
boolOp = fr'(?P<boolOp>\.AND|\.XOR|\.OR)'
imadSem  = fr'(?:\.MOV|\.SHL|\.IADD)?'
imadType = fr'(?P<type>\.U32|\.S32)?'
cmpType = fr'(?P<type>\.U32|\.S32)?'
hmmaShape = fr'(?P<shape>\.1688|\.16816)'
hmmaType = fr'(?P<type>\.F16|\.F32)'
mma884 = fr'(?P<dtype>\.F16|\.F32)(?P<ctype>\.F16|\.F32)(?P<step>\.STEP0|\.STEP1|\.STEP2|\.STEP3)'
immaInfix = fr'(?P<infix>\.8816|\.8832)'
immaT0 = fr'(?P<type0>\.U8|\.S8|\.U4|\.S4)'
immaT1 = fr'(?P<type1>\.U8|\.S8|\.U4|\.S4)'
shf  = fr'(?:(?P<lr>\.L|\.R)(?P<type>\.S32|\.U32|\.S64|\.U64)?(?P<hi>\.HI)?)'
shfl = fr'(?P<shfl>\.IDX|\.UP|\.DOWN|\.BFLY)'
atomType = fr'(?P<atom>\.ADD|\.DEC|\.EXCH)'
frnd = fr'(?P<frnd>\.RN|\.RZ|\.RM|\.RP)?'
irnd = fr'(?P<irnd>\.TRUNC|\.FLOOR|\.CEIL)?(?P<ntz>\.NTZ)?'
# >= Turing
ldsmLayout = fr'(?P<ldsmLayout>\.M88|\.MT88)'
numMat = fr'(?P<numMat>\.1|\.2|\.4)'
depbarCmp = fr'(?P<depbarCmp>\.LE)'



grammar = {
  # Currently, we believe 9 bits are used to index instructions.
  # Or 12 bits, with 3 bits to be modifed (icr).

  # Movement instructions
  'MOV' : [{'code' : 0x202, 'rule' : rf'MOV {rd}, {icrs1};', 'lat' : 5}],

  # Load/Store instructions 
  'LDG' : [{'code' : 0x381, 'rule' : rf'LDG{memType}{memCache}{memStrong}{memScope}\s*{ldgp} {rd}, {addr24};' },
           {'code' : 0x981, 'rule' : rf'LDG{memType}{memCache}{memStrong}{memScope}\s*{ldgp} {rd}, {addr24ld};'}],
  'STG' : [{'code' : 0x386, 'rule' : rf'STG{memType}{memCache}{memScope}{memStrong} {addr24}, {rs1};'}],
  'LDS' : [{'code' : 0x984, 'rule' : rf'LDS{memType}{memCache} {rd}, {addr24};'}],
  'STS' : [{'code' : 0x388, 'rule' : rf'STS{memType}{memCache} {addr24st}, {rs1};'}],
  # Load matrix
  # TODO: UR offset
  'LDSM' : [{'code' : 0x83b, 'rule' : rf'LDSM.16{ldsmLayout}{numMat} {rd}, {addr24};'}],
  # TODO: .ZFILL & UR offset
  'LDGSTS' : [{'code' : 0xfae, 'rule' : rf'LDGSTS{ldgsts} {addrLdgstsDst}, {addrLdgstsSrc}{ps0i};'}],
  'LDGDEPBAR' : [{'code' : 0x9af, 'rule' : rf'LDGDEPBAR;'}],
  'DEPBAR' : [{'code' : 0x91a, 'rule' : rf'DEPBAR{depbarCmp} SB0, {depbarN};'}],
  # LDC has its own rule.
  'LDC' : [{'code' : 0xb82, 'rule' : rf'LDC{memType} {rd}, {cs1};'}], # Add register offset.

  # Uniform datapath instructions
  'UMOV' : [{'code' : 0x282, 'rule' : rf'UMOV {ud}, {icrus1};'}],
  'USHF' : [{'code' : 0x299, 'rule' : rf'USHF{shf} {ud}, {us0}, {icrus1}, {us2};', 'lat' : 5}],
  'UIADD3':[{'code' : 0x290, 'rule' : rf'UIADD3{X} {ud}, {upd0i}{upd1i}{us0}, {icrus1}, {us2}{ups0i}{ups1i};', 'lat' : 4},
            {'code' : 0x290, 'rule' : rf'UIADD3{X} {ud}, {upd0i}{upd1i}{us0}, {us2}, {ic2}{ups0i}{ups1i};'}],
  'ULEA'  :[{'code' : 0x291, 'rule' : rf'ULEA(?P<hi>\.HI)?{X}(?P<sx32>\.SX32)? {ud}, {upd0i}{us0}, {icrus1}, {is11w5}{ups0i}{ups1i};'},
            {'code' : 0x291, 'rule' : rf'ULEA(?P<hi>\.HI)?{X}(?P<sx32>\.SX32)? {ud}, {upd0i}{us0}, {icrus1}, {us2}, {is11w5}{ups0i}{ups1i};'}],

  'ULDC' : [{'code' : 0xab9, 'rule' : rf'ULDC{memType} {ud}, {cs1};'}],

  # Integer instructions
  'IADD3': [{'code' : 0x210, 'rule' : rf'IADD3{X} {rd}, {pd0i}{pd1i}{rs0}, {icrus1}, {rs2}{ps0i}{ps1i};', 'lat' : 4},
            {'code' : 0x210, 'rule' : rf'IADD3{X} {rd}, {pd0i}{pd1i}{rs0}, {rs2}, {ic2}{ps0i}{ps1i};'}],
  'IMUL' : [{'code' : 0x000, 'rule' : r'IMUL;'}],
  'LEA'  : [{'code' : 0x211, 'rule' : rf'LEA(?P<hi>\.HI)?{X}(?P<sx32>\.SX32)? {rd}, {pd0i}{rs0}, {icrus1}, {is11w5}{ps0i}{ps1i};'},
            {'code' : 0x211, 'rule' : rf'LEA(?P<hi>\.HI)?{X}(?P<sx32>\.SX32)? {rd}, {pd0i}{rs0}, {icrus1}, {rs2}, {is11w5}{ps0i}{ps1i};'}],
  # Do not capture them () as flags. But treat them as different instructions.
  'IMAD'  : [{'code' : 0x224, 'rule' : rf'IMAD{imadSem}{imadType}{X} {rd}, {pd0i}{rs0}, {icrs1}, {rs2}{ps0i};', 'lat' : 5},
             {'code' : 0x224, 'rule' : rf'IMAD{imadSem}{imadType}{X} {rd}, {pd0i}{rs0}, {rs2}, {ic2}{ps0i};', 'lat' : 5},
             {'code' : 0x224, 'rule' : rf'IMAD{imadSem}{imadType}{X} {rd}, {pd0i}{rs0}, {rs2}, {us2e1}{ps0i};', 'lat' : 5}, 
             {'code' : 0x225, 'rule' : rf'IMAD.WIDE{imadType} {rd}, {pd0i}{rs0}, {icrs1}, {rs2}{ps0i};'}, 
             {'code' : 0x225, 'rule' : rf'IMAD.WIDE{imadType} {rd}, {pd0i}{rs0}, {rs2}, {ic2}{ps0i};'}, 
             # IMAD.HI is special. rs2 represent register after it. Must be even register.
             {'code' : 0x227, 'rule' : rf'IMAD.HI{imadType} {rd}, {pd0i}{rs0}, {icrs1}, {rs2}{ps0i};'}], 
  'ISETP' : [{'code' : 0x20c, 'rule' : rf'ISETP{icmp}{cmpType}{boolOp} {pd0}, {pd1}, {rs0}, {icrus1}, {ps0};', 'lat' : 4}],
  'LOP3'  : [{'code' : 0x212, 'rule' : rf'LOP3\.LUT {pd0i}{rd}, {rs0}, {icrus1}, {rs2}, {isw8}(?:, {ps0})?;', 'lat' : 5}],
  'SHF'   : [{'code' : 0x219, 'rule' : rf'SHF{shf} {rd}, {rs0}, {icrs1}, {rs2};', 'lat' : 5}], # Somethings 4. st. 5.
  'PRMT'  : [{'code' : 0x216, 'rule' : rf'PRMT {rd}, {rs0}, {icrs1}, {rs2};', 'lat': 5}],
  'SEL'   : [{'code' : 0x207, 'rule' : rf'SEL {rd}, {rs0}, {icrs1}, {ps0};', 'lat': 4}],
  'SGXT'  : [{'code' : 0x21a, 'rule' : rf'SGXT{imadType} {rd}, {rs0}, {icrs1};', 'lat': 5}],
  'IABS'  : [{'code' : 0x213, 'rule' : rf'IABS {rd}, {icrs1};', 'lat': 6}],
  'IMNMX' : [{'code' : 0x217, 'rule' : rf'IMNMX{imadType} {rd}, {rs0}, {icrs1}, {ps0};'}],

  # Half instructions
  'HADD2' : [{'code' : 0x230, 'rule' : rf'HADD2 {rd}, {rs0}, {rs1};', 'lat' : 7}],
  'HFMA2' : [{'code' : 0x231, 'rule' : rf'HFMA2 {rd}, {rs0}, {rs1}, {rs2};', 'lat' : 7}],
  'HMUL2' : [{'code' : 0x232, 'rule' : rf'HMUL2 {rd}, {rs0}, {rs1};', 'lat' : 7}],

  # Float instructions
  'FFMA' : [{'code' : 0x223, 'rule' : rf'FFMA {rd}, {rs0}, {fcrs1}, {rs2};'}, 
            {'code' : 0x223, 'rule' : rf'FFMA {rd}, {rs0}, {rs2}, {fc2};'}],
  'MUFU' : [{'code' : 0x308, 'rule' : rf'MUFU{mufu} {rd}, {fcrs1};'},],
  # FADD has its own rule. 
  'FADD' : [{'code' : 0x221, 'rule' : rf'FADD {rd}, {rs0}, {fcrs1add};'}],
  'FMUL' : [{'code' : 0x220, 'rule' : rf'FMUL {rd}, {rs0}, {fcrs1};'}],
  'FMNMX': [{'code' : 0x209, 'rule' : rf'FMNMX {rd}, {rs0}, {rs1}, {ps0};'}], # ps0=PT:fmin
  # TensorCore instructions
  'HMMA' : [{'code' : 0x23c, 'rule' : rf'HMMA{hmmaShape}{hmmaType} {rd}, {rs0}, {rs1}, {rs2};'},
            {'code' : 0x236, 'rule' : rf'HMMA\.884{mma884} {rd}, {rs0mma884}, {rs1mma884}, {rs2};'}],
  'IMMA' : [{'code' : 0x237, 'rule' : rf'IMMA{immaInfix}{immaT0}{immaT1} {rd}, {rs0}\.ROW, {rs1}(?P<s1col>\.COL), {rs2};'},],
  'BMMA' : [{'code' : 0x23d, 'rule' : rf'BMMA\.88128(?P<bmma>\.POPC) {rd}, {rs0}\.ROW, {rs1}(?P<s1col>\.COL), {rs2};'},],
  # Control instructions
  'BRA'  : [{'code' : 0x947, 'rule' : rf'BRA((?P<bra>\.U))? {cp}{is1};', 'lat' : 7}], # Lat?
  'BSSY' : [{'code' : 0x945, 'rule' : rf'BSSY {sb}, {cp}{is1};'}],
  'BSYNC': [{'code' : 0x941, 'rule' : rf'BSYNC {cp}{sb};'}],
  'BMOV' : [{'code' : 0x355, 'rule' : rf'BMOV.32.CLEAR {rd}, {sb};'}],
  'EXIT' : [{'code' : 0x94d, 'rule' : rf'EXIT\s*{cp};'}], 
  # Miscellaneous instructions.
  'CS2R' : [{'code' : 0x805, 'rule' : fr'CS2R {rd}, {sr};', 'lat' : 5}],
  'S2R'  : [{'code' : 0x919, 'rule' : fr'S2R {rd}, {sr};', 'lat' : 7}],  
  'S2UR'  : [{'code' : 0x9c3, 'rule' : fr'S2UR {ud}, {sr};', 'lat': 20}],
  'NOP'  : [{'code' : 0x918, 'rule' : r'NOP;'}],
  'YIELD': [{'code' : 0x946, 'rule' : fr'YIELD;'}], # & ps0 = pt
  'BAR'  : [{'code' : 0xb1d, 'rule' : fr'BAR(?P<bar>\.SYNC|\.SYNC.DEFER_BLOCKING|) {bar};'}], 
  # Predicate instructions.
  'P2R' : [{'code' : 0x803, 'rule' : fr'P2R {rd}, (?P<pr>PR), ({rs0}, )?{is1};', 'lat' : 8}],
  'R2P' : [{'code' : 0x804, 'rule' : fr'R2P PR, {rs0}, {is1};', 'lat' : 12}],
  # Shuffle
  'SHFL' : [{'code' : 0x389, 'rule' : fr'SHFL{shfl} {pd0}, {rd}, {rs0}, {icrs1}, {shflMask};', 'lat' : 20}],
  # Conversion
  'F2F' : [{'code' : 0x304, 'rule' : fr'F2F(?P<dtype>\.F16|\.F32)(?P<stype>\.F16|\.F32) {rd}, {rs1};', 'lat':5}],
  'F2FP': [{'code' : 0x23e, 'rule' : fr'F2FP(?P<relu>\.RELU)?(?P<dtype>\.BF16)?\.PACK_AB {rd}, {rs0}, {fcrs1};'}],
  'F2I' : [{'code' : 0x305, 'rule' : fr'F2I{f2i32}{irnd} {rd}, {rs1};'}, 
           # 64-bit
           {'code' : 0x311, 'rule' : fr'F2I{f2i64}{irnd} {rd}, {rs1};'}],

  'I2F' : [{'code' : 0x306, 'rule' : fr'I2F{i2f32}{frnd} {rd}, {rs1};'}, 
           {'code' : 0x312, 'rule' : fr'I2F(i2f64){frnd} {rd}, {rs1};'}],
  # Atomic instructions
  # TODO: UR offset
  'ATOMS' : [{'code' : 0x38d, 'rule' : fr'ATOMS.CAS {rd}, {addr24}, {rs1}, {rs2};'},
             {'code' : 0x38c, 'rule' : fr'ATOMS{atomType} {rd}, {addr24}, {rs1};'}],
  'ATOMG' : [{'code' : 0x3a9, 'rule' : fr'ATOMG.CAS{memStrong}{memScope} {rd}, {addr24}, {rs1}, {rs2};'},
              {'code' : 0x3a8, 'rule' : fr'ATOMG.EXCH{memStrong}{memScope} {rd}, {addr24}, {rs1};'}],
  # with UR offset [R2+UR4+0x4] => bits[27:26] = 0b11
  'RED' : [{'code' : 0x98e, 'rule' : fr'RED{redOp}{memCache}{redType}{memStrong}{memScope} {addr24ld}, {rs1};'}],
}



flag_str = '''
LDG, STG, LDS, STS, LDC, ULDC, LDGSTS: type
0<<9 .U8
1<<9 .S8
2<<9 .U16
3<<9 .S16
4<<9 DEFAULT
4<<9 .32
5<<9 .64
6<<9 .128

LDG, STG, ATOMS, ATOMG, RED: strong
0<<15 .CONSTANT
1<<15 DEFAULT
2<<15 .STRONG
3<<15 .WEEK     # FIXME

LDG, STG, RED: cache
0<<20 .EF
1<<20 DEFAULT
3<<20 .LU

LDG, RED: off64
1<<26 .64

LDS: U
1<<12 .U

LDG, STG, RED: E
1<<8 .E

LDG, STG, ATOMG, RED: scope
0<<13 .CTA
2<<13 .GPU
3<<13 DEFAULT
3<<13 .SYS

LDSM: ldsmLayout
0<<12 .M88
4<<12 .MT88

LDSM: numMat
0<<8 .1
1<<8 .2
2<<8 .4

LDGSTS: ltc128b # other LDST use bit[8] to encode .E
1<<8 .LTC128B

BRA: is1neg
262143<<0 - # 0x3ffff

S2R, S2UR: sr
0<<8 SR_LANEID
33<<8 SR_TID.X
34<<8 SR_TID.Y
35<<8 SR_TID.Z
37<<8 SR_CTAID.X
38<<8 SR_CTAID.Y
39<<8 SR_CTAID.Z

CS2R: sr
80<<8 SR_CLOCKLO
511<<8 SRZ

IMAD, IMNMX: type
0<<0 .U32
1<<9 DEFAULT
1<<9 .S32 
# .U16
# .S16
# .U64
# .S64

RED: redOp
0<<23 .ADD
1<<23 .MIN
2<<23 .MAX
5<<23 .AND


RED: redType
0<<9 DEFAULT
0<<9 .U32
1<<9 .S32
3<<9 .F32
4<<9 .F16x2.RN
6<<9 .F64.FTZ.RN

ISETP: cmp
1<<12 .LT
2<<12 .EQ
3<<12 .LE
4<<12 .GT
5<<12 .NE
6<<12 .GE

ISETP: boolOp
0<<0  .AND 
1<<10 .OR
1<<11 .XOR

ISETP: type
1<<9 .S32
1<<9 DEFAULT
0<<9 .U32

P2R: pr
255<<88 PR

SHF, USHF: lr
0<<12 .L
1<<12 .R

SHF, USHF: type
0<<8 .S64
0<<8 .U64
4<<8 .S32
6<<8 .U32
6<<8 DEFAULT

SHFL: shfl
0<<122 .IDX
1<<122 .UP
2<<122 .DOWN
3<<122 .BFLY

SHF, LEA, ULEA: hi
1<<16 .HI

IADD3, UIADD3, LOP3, IMAD, LEA, ULEA: nopd0
7<<17

IADD3, UIADD3: nopd1
7<<20

IADD3, UIADD3, LOP3, IMAD, LEA, ULEA, LDGSTS: nops0
15<<23

IADD3, UIADD3: nops1
15<<13

IADD3, UIADD3, LEA, ULEA, IMAD: x
1<<10 .X


FADD, IADD3: rs1neg
1<<127 -

UIADD3: us1neg
1<<127 -

FADD, IADD3: rs0neg
1<<8 -

UIADD3: us0neg
1<<8 -

IMAD: rs2neg
1<<11 -

LEA, ULEA: sx32
1<<9 .SX32

MUFU: mufu
0<<10 .COS
1<<10 .SIN
2<<10 .EX2
3<<10 .LG2
4<<10 .RCP
5<<10 .RSQ
6<<10 .RCP64H
7<<10 .RSQ64H
8<<10 .SQRT

##################################################
# tensor-core flags
HMMA: type
0<<0 .F16
1<<12 .F32

# HMMA: abtype
# 0<<0 #default
# 8<<16 .TF32
# 4<<16 .BF16

HMMA: shape
0<<0 .1688
1<<11 .16816

HMMA: rs0layout
0<<9 .ROW
1<<9 .COL

HMMA: rs1layout
0<<10 .ROW
1<<10 .COL

HMMA: dtype
0<<12 .F16
1<<12 .F32

HMMA: ctype
0<<14 .F16
1<<14 .F32

HMMA: step
0<<15 .STEP0
1<<15 .STEP1
2<<15 .STEP2
3<<15 .STEP3

IMMA: type0
0<<12 .U8
0<<12 .U4
1<<12 .S8
1<<12 .S4

IMMA: type1
0<<14 .U8
0<<14 .U4
1<<14 .S8
1<<14 .S4

IMMA: infix
0<<16 .8816
56<<16 .8832

IMMA, BMMA: s1col
1<<10 .COL

BMMA: bmma
1<<16 .POPC

BRA: u
1<<96 .U

BAR: bar
0<<0 .SYNC
1<<13 .ARV
1<<14 .RED.POPC
1<<15 .SYNCALL
1<<16 .SYNC.DEFER_BLOCKING

# type conversions
F2F: dtype
1<<21 .F16

F2F: stype
1<<11 .F32

F2FP: type
0<<16 DEFAULT
1<<16 .BF16

F2FP: relu
1<<16 .RELU

I2F: dtype
1<<11 .F16
2<<11 DEFAULT
2<<11 .F32
3<<11 .F64

I2F: swidth
0<<20 8
1<<20 16
2<<20 DEFAULT
2<<20 32
3<<20 64

I2F: ssign
0<<10 .U
1<<10 DEFAULT
1<<10 .S

I2F: frnd
0<<14 DEFAULT
0<<14 .RN
1<<14 .RM
2<<14 .RP
3<<14 .RZ

F2I: irnd
0<<14 DEFAULT
1<<14 .FLOOR
2<<14 .CEIL
3<<14 .TRUNC

F2I: ftz
1<<16 .FTZ

F2I: ntz
1<<13 .NTZ

F2I: dsign
0<<8 .U
1<<8 DEFAULT
1<<8 .S

F2I: dwidth
0<<11 8
1<<11 16
2<<11 DEFAULT
2<<11 32
3<<11 64

F2I: stype
1<<20 .F16
2<<20 DEFAULT
2<<20 .F32
3<<20 .F64


# sm75+
DEPBAR: depbarCmp
1<<111 .LE
'''

# Create flag dict
flags = {}
for key in grammar.keys():
  flags[key] = {}

# Format flag_str
# Delete comments
flag_str = re.sub(r'#.*?\n', '', flag_str)
flag_str = re.sub(r'\n\n', '\n', flag_str)
flag_str = re.sub(r'^\n', '', flag_str)
flag_str = re.sub(r'\n$', '', flag_str)
flag_str = re.sub(r'\s+\n', '\n', flag_str)

for line in flag_str.split('\n'):
  flag_line = re.match(r'(\d+)<<(\d+)\s*(.*)?', line)
  if flag_line:
    value = int(flag_line.group(1))<<int(flag_line.group(2))
    for op in ops:
      flags[op][name][flag_line.group(3)] = value
  else:
    ops, name = re.split(r':\s*', line)
    ops = re.split(r',\s*', ops)
    # Create new dict for this flag.
    for op in ops:
      try:
        flags[op][name] = {}
      except Exception as e:
        print(ops)
        raise e



ctrl_re = r'(?P<ctrl>[0-9a-fA-F\-]{2}:[0-5\-]:[0-5\-]:[\-yY]:[0-9a-fA-F\-])'
pred_re = r'(?P<pred>@(?P<predNot>!)?P(?P<predReg>\d)\s+)'
inst_re = fr'{pred_re}?(?P<op>\w+)(?P<rest>[^;]*;)'

def ReadCtrl(ctrl, gram):
  # Input should be '--:-:-:-:2'
  # Return a bitstring/hex.
  # Not include reuse flag.
  watdb, readb, wrtdb, yield_, stall = ctrl.split(":")
  watdb = 0 if watdb == '--' else int(watdb)
  readb = 7 if readb == '-' else int(readb)
  wrtdb = 7 if wrtdb == '-' else int(wrtdb)
  yield_ = 0 if yield_ == 'y' or yield_ == 'Y' else 1
  try: 
    stall = int(stall, 16)
  except ValueError:
    if 'lat' in gram:
      stall = gram['lat']
    else:
      stall = 1 # 1 is the minimum pipeline stall
  return stall | yield_ << 4 | wrtdb << 5 | readb << 8 | watdb << 11

def GenReuse(captured_dict):
  reuse_code = 0x0
  if captured_dict.get('reuse1') == '.reuse':
    reuse_code |= 0x1 << 58
  if captured_dict.get('reuse2') == '.reuse':
    reuse_code |= 0x1 << 59
  if captured_dict.get('reuse3') == '.reuse':
    reuse_code |= 0x1 << 60
  return reuse_code

def ProcessAsmLine(line, line_num):
  result = re.match(fr'^{ctrl_re}(?P<space>\s+){inst_re}', line)
  if result != None:
    result = result.groupdict()
    return {
      'line_num' : line_num,
      'pred'     : result['pred'], # TODO: None?
      'predNot'  : result['predNot'],
      'predReg'  : result['predReg'],
      'space'    : result['space'],
      'op'       : result['op'],
      'rest'     : result['rest'], # TODO: Add instr. 
      'ctrl'     : result['ctrl']
    }
  else:
    return None # TODO: raise Exception?


icr_dict = {
  'base' : 0x0,
  'is1'  : 0x8,
  'fs1'  : 0x8,
  'fs1add' : 0x4,
  'cs1'  : 0xa,
  'cs1add' : 0x6,
  'is2'  : 0x4,
  'cs2'  : 0x6,
  'fs2'  : 0x4,
  'us1'  : 0xc,
  'us2e1'  : 0xe, # IMAD
  'uoffld' : 0x9, # LDG
  'uoffst' : 0x9, # STS
}

def GenCode(op, gram, captured_dict, asm_line):
  '''
  asm_line:
    Result of ProcessAsmLine 
      dict = {line_num, pred, predNot, predReg, space, op, rest, ctrl}
  128 bits in total:

  First 64 bits:
    registers, pred, instruction.
  Second 64 bits:
    23 bits for control information. Rest for flags/3rd-reigster.

  Return:
    code (0xffff..ff)
  '''
  # Start with instruction code.
  try:
    code = gram['code'] << 64 
  except Exception as e:
    print(asm_line)
    raise e
  flag = flags[op]

  # Process predicate.
  p = int(asm_line['predReg']) if asm_line['predReg'] else 7 # 7 => PT
  if asm_line['predNot']:
    p |= 0x8
  code |= p << (64+12) # 12 bits for instruction encoding.

  # Reuse flag.
  code |= GenReuse(captured_dict)

  # Captured data.
  #   icr? (operand)
  #   flags. (part of op)
  # Change to different version of op code.
  for key in icr_dict.keys():
    if key in captured_dict:
      if captured_dict[key] != None:
        if 'ud' in captured_dict and key in ['us1', 'us2'] \
                                 and asm_line['op'] != 'UMOV': # skip u-instr
          continue
        code &= ~(1<<(64+9))
        code |= icr_dict[key] << 72

  # Special rules for ur
  for key in captured_dict:
    if key in ['us0', 'us1', 'us2', 'us2e1', 'uoffld', 'uoffst'] and captured_dict[key] != None:
      # if not 'ud' in key:
      code |= 0x1<<27

  # Operands
  for key, value in captured_dict.items():
    if key in flag:
      # Take care of default value
      if value == None:
        if 'DEFAULT' in flag[key]:
          flag_value = flag[key]['DEFAULT']
          code |= flag_value
          continue
        else:
          continue
      flag_value = flag[key][value]
      code |= flag_value
    if value == None: # TODO: better way to do this?
      continue
    if key in operands.keys():
      code |= operands[key](value)
    
  # Control flag.
  code |= ReadCtrl(asm_line['ctrl'], gram) << 41

  # TODO: Ideally, they should be deleted. (For what ever reason, they exist.)
  # End rules.
  if op == 'MOV':
    code |= 0xf00
  if op == 'ISETP':
    code |= 0x7 << 4
  if op == 'LDC':
    code |= 0xff << 88
  if op == 'LDGSTS':
    code |= 0x8101c48
  if op == 'BMOV':
    code |= 0x100000
  if op == 'LEA': # LEA has RZ as default operand
    code |= 0xff
  if op == 'ULEA':
    code |= 0x3f
  if op == 'F2FP':
    code |= 0xff
  if op == 'RED':
    code |= 0x0
  # if op == 'BRA':
  #  code |= 0x1 << 96

  # if op == 'LDG':
  #   print(hex(code))

  return code

