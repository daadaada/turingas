"""Independent NVIDIA-generated SHFL encoding regression fixtures.

Generated from fixtures/shfl_sm75.ptx using ptxas 12.9.86 and nvdisasm 12.9.88:
  ptxas -arch=sm_70 tests/fixtures/shfl_sm75.ptx -o shfl_sm70.cubin
  nvdisasm -c -hex shfl_sm70.cubin
Repeat with sm_75/sm_80. Original compile/disassembly produced 24 relevant
instructions per architecture: 4 modes times 2 lane forms times 3 clamp forms
(register, immediate 0x1f and segmented immediate 0x1f1f). Data input depends
on thread ID so shuffled output cannot be discarded as uniform. SM70 also
inserts inactive SHFL instructions; those are excluded from this corpus.

Expected instruction/control words come from NVIDIA output, not this
assembler. No GPU or CUDA install is needed for these regressions.
"""
import re
import unittest
from turingas.grammar import GenCode, ProcessAsmLine, grammar

# (SM, disassembly, instruction low word, instruction high word)
FIXTURES = [
    (70, 'SHFL.DOWN P0, R5, R0, 0x7, 0x1f1f;', 0x08ff1f0000057f89, 0x000e240000000000),
    (70, 'SHFL.DOWN P0, R5, R0, 0x7, 0x1f;', 0x08e01f0000057f89, 0x000e240000000000),
    (70, 'SHFL.DOWN P0, R5, R0, 0x7, R5;', 0x08e0000000057989, 0x000e240000000005),
    (70, 'SHFL.DOWN P0, R5, R0, R5, 0x1f1f;', 0x081f1f0500057589, 0x000e240000000000),
    (70, 'SHFL.DOWN P0, R5, R0, R5, 0x1f;', 0x08001f0500057589, 0x000e240000000000),
    (70, 'SHFL.DOWN P0, R5, R0, R5, R4;', 0x0800000500057389, 0x000e240000000004),
    (70, 'SHFL.UP P0, R5, R0, 0x7, 0x1f1f;', 0x04ff1f0000057f89, 0x000e240000000000),
    (70, 'SHFL.UP P0, R5, R0, 0x7, 0x1f;', 0x04e01f0000057f89, 0x000e240000000000),
    (70, 'SHFL.UP P0, R5, R0, 0x7, R5;', 0x04e0000000057989, 0x000e240000000005),
    (70, 'SHFL.UP P0, R5, R0, R5, 0x1f1f;', 0x041f1f0500057589, 0x000e240000000000),
    (70, 'SHFL.UP P0, R5, R0, R5, 0x1f;', 0x04001f0500057589, 0x000e240000000000),
    (70, 'SHFL.UP P0, R5, R0, R5, R4;', 0x0400000500057389, 0x000e240000000004),
    (70, 'SHFL.BFLY P0, R5, R0, 0x7, 0x1f1f;', 0x0cff1f0000057f89, 0x000e240000000000),
    (70, 'SHFL.BFLY P0, R5, R0, 0x7, 0x1f;', 0x0ce01f0000057f89, 0x000e240000000000),
    (70, 'SHFL.BFLY P0, R5, R0, 0x7, R5;', 0x0ce0000000057989, 0x000e240000000005),
    (70, 'SHFL.BFLY P0, R5, R0, R5, 0x1f1f;', 0x0c1f1f0500057589, 0x000e240000000000),
    (70, 'SHFL.BFLY P0, R5, R0, R5, 0x1f;', 0x0c001f0500057589, 0x000e240000000000),
    (70, 'SHFL.BFLY P0, R5, R0, R5, R4;', 0x0c00000500057389, 0x000e240000000004),
    (70, 'SHFL.IDX P0, R5, R0, 0x7, 0x1f1f;', 0x00ff1f0000057f89, 0x000e240000000000),
    (70, 'SHFL.IDX P0, R5, R0, 0x7, 0x1f;', 0x00e01f0000057f89, 0x000e240000000000),
    (70, 'SHFL.IDX P0, R5, R0, 0x7, R5;', 0x00e0000000057989, 0x000e240000000005),
    (70, 'SHFL.IDX P0, R5, R0, R5, 0x1f1f;', 0x001f1f0500057589, 0x000e240000000000),
    (70, 'SHFL.IDX P0, R5, R0, R5, 0x1f;', 0x00001f0500057589, 0x000e240000000000),
    (70, 'SHFL.IDX P0, R5, R0, R5, R4;', 0x0000000500057389, 0x000e240000000004),
    (75, 'SHFL.DOWN P0, R0, R0, 0x7, 0x1f1f;', 0x08ff1f0000007f89, 0x000e240000000000),
    (75, 'SHFL.DOWN P0, R0, R0, 0x7, 0x1f;', 0x08e01f0000007f89, 0x000e240000000000),
    (75, 'SHFL.DOWN P0, R0, R0, 0x7, R3;', 0x08e0000000007989, 0x000e240000000003),
    (75, 'SHFL.DOWN P0, R0, R0, R3, 0x1f1f;', 0x081f1f0300007589, 0x000e240000000000),
    (75, 'SHFL.DOWN P0, R0, R0, R3, 0x1f;', 0x08001f0300007589, 0x000e240000000000),
    (75, 'SHFL.DOWN P0, R0, R0, R3, R4;', 0x0800000300007389, 0x000e240000000004),
    (75, 'SHFL.UP P0, R0, R0, 0x7, 0x1f1f;', 0x04ff1f0000007f89, 0x000e240000000000),
    (75, 'SHFL.UP P0, R0, R0, 0x7, 0x1f;', 0x04e01f0000007f89, 0x000e240000000000),
    (75, 'SHFL.UP P0, R0, R0, 0x7, R3;', 0x04e0000000007989, 0x000e240000000003),
    (75, 'SHFL.UP P0, R0, R0, R3, 0x1f1f;', 0x041f1f0300007589, 0x000e240000000000),
    (75, 'SHFL.UP P0, R0, R0, R3, 0x1f;', 0x04001f0300007589, 0x000e240000000000),
    (75, 'SHFL.UP P0, R0, R0, R3, R4;', 0x0400000300007389, 0x000e240000000004),
    (75, 'SHFL.BFLY P0, R0, R0, 0x7, 0x1f1f;', 0x0cff1f0000007f89, 0x000e240000000000),
    (75, 'SHFL.BFLY P0, R0, R0, 0x7, 0x1f;', 0x0ce01f0000007f89, 0x000e240000000000),
    (75, 'SHFL.BFLY P0, R0, R0, 0x7, R3;', 0x0ce0000000007989, 0x000e240000000003),
    (75, 'SHFL.BFLY P0, R0, R0, R3, 0x1f1f;', 0x0c1f1f0300007589, 0x000e240000000000),
    (75, 'SHFL.BFLY P0, R0, R0, R3, 0x1f;', 0x0c001f0300007589, 0x000e240000000000),
    (75, 'SHFL.BFLY P0, R0, R0, R3, R4;', 0x0c00000300007389, 0x000e240000000004),
    (75, 'SHFL.IDX P0, R0, R0, 0x7, 0x1f1f;', 0x00ff1f0000007f89, 0x000e240000000000),
    (75, 'SHFL.IDX P0, R0, R0, 0x7, 0x1f;', 0x00e01f0000007f89, 0x000e240000000000),
    (75, 'SHFL.IDX P0, R0, R0, 0x7, R3;', 0x00e0000000007989, 0x000e240000000003),
    (75, 'SHFL.IDX P0, R0, R0, R3, 0x1f1f;', 0x001f1f0300007589, 0x000e240000000000),
    (75, 'SHFL.IDX P0, R0, R0, R3, 0x1f;', 0x00001f0300007589, 0x000e240000000000),
    (75, 'SHFL.IDX P0, R0, R0, R3, R4;', 0x0000000300007389, 0x000e240000000004),
    (80, 'SHFL.DOWN P0, R5, R0, 0x7, 0x1f1f;', 0x08ff1f0000057f89, 0x000e240000000000),
    (80, 'SHFL.DOWN P0, R5, R0, 0x7, 0x1f;', 0x08e01f0000057f89, 0x000e240000000000),
    (80, 'SHFL.DOWN P0, R5, R0, 0x7, R5;', 0x08e0000000057989, 0x000e240000000005),
    (80, 'SHFL.DOWN P0, R5, R0, R5, 0x1f1f;', 0x081f1f0500057589, 0x000e240000000000),
    (80, 'SHFL.DOWN P0, R5, R0, R5, 0x1f;', 0x08001f0500057589, 0x000e240000000000),
    (80, 'SHFL.DOWN P0, R5, R0, R5, R4;', 0x0800000500057389, 0x000e240000000004),
    (80, 'SHFL.UP P0, R5, R0, 0x7, 0x1f1f;', 0x04ff1f0000057f89, 0x000e240000000000),
    (80, 'SHFL.UP P0, R5, R0, 0x7, 0x1f;', 0x04e01f0000057f89, 0x000e240000000000),
    (80, 'SHFL.UP P0, R5, R0, 0x7, R5;', 0x04e0000000057989, 0x000e240000000005),
    (80, 'SHFL.UP P0, R5, R0, R5, 0x1f1f;', 0x041f1f0500057589, 0x000e240000000000),
    (80, 'SHFL.UP P0, R5, R0, R5, 0x1f;', 0x04001f0500057589, 0x000e240000000000),
    (80, 'SHFL.UP P0, R5, R0, R5, R4;', 0x0400000500057389, 0x000e240000000004),
    (80, 'SHFL.BFLY P0, R5, R0, 0x7, 0x1f1f;', 0x0cff1f0000057f89, 0x000e240000000000),
    (80, 'SHFL.BFLY P0, R5, R0, 0x7, 0x1f;', 0x0ce01f0000057f89, 0x000e240000000000),
    (80, 'SHFL.BFLY P0, R5, R0, 0x7, R5;', 0x0ce0000000057989, 0x000e240000000005),
    (80, 'SHFL.BFLY P0, R5, R0, R5, 0x1f1f;', 0x0c1f1f0500057589, 0x000e240000000000),
    (80, 'SHFL.BFLY P0, R5, R0, R5, 0x1f;', 0x0c001f0500057589, 0x000e240000000000),
    (80, 'SHFL.BFLY P0, R5, R0, R5, R4;', 0x0c00000500057389, 0x000e240000000004),
    (80, 'SHFL.IDX P0, R5, R0, 0x7, 0x1f1f;', 0x00ff1f0000057f89, 0x000e240000000000),
    (80, 'SHFL.IDX P0, R5, R0, 0x7, 0x1f;', 0x00e01f0000057f89, 0x000e240000000000),
    (80, 'SHFL.IDX P0, R5, R0, 0x7, R5;', 0x00e0000000057989, 0x000e240000000005),
    (80, 'SHFL.IDX P0, R5, R0, R5, 0x1f1f;', 0x001f1f0500057589, 0x000e240000000000),
    (80, 'SHFL.IDX P0, R5, R0, R5, 0x1f;', 0x00001f0500057589, 0x000e240000000000),
    (80, 'SHFL.IDX P0, R5, R0, R5, R4;', 0x0000000500057389, 0x000e240000000004),
]


def control_text(high):
    """Render NVIDIA dependency bits in TuringAs control syntax."""
    control = high >> 41
    stall, yield_bit = control & 15, (control >> 4) & 1
    write, read, wait = (control >> 5) & 7, (control >> 8) & 7, (control >> 11) & 63
    return "%s:%s:%s:%s:%x" % (
        "%02d" % wait if wait else "--",
        "-" if read == 7 else str(read),
        "-" if write == 7 else str(write),
        "-" if yield_bit else "Y", stall)


class ShflEncodingTests(unittest.TestCase):
    def test_immediates_outside_encoding_fields_are_not_truncated(self):
        from turingas.turas import assemble
        for lane, clamp in (("0x20", "0x1f"), ("R3", "0x2000")):
            with self.subTest(lane=lane, clamp=clamp):
                with self.assertRaises(ValueError):
                    assemble("--:-:-:-:2 SHFL.IDX PT, R5, R0, %s, %s;" % (lane, clamp))

    def test_register_clamp_is_counted_in_kernel_resources(self):
        from turingas.turas import assemble
        kernel = assemble("--:-:-:-:2 SHFL.IDX PT, R5, R0, R3, R200;")
        self.assertGreaterEqual(kernel["RegCnt"], 201)

    def test_matches_independent_sm70_sm75_sm80_words(self):
        for sm, instruction, low, high in FIXTURES:
            with self.subTest(sm=sm, instruction=instruction):
                line = ProcessAsmLine(control_text(high) + " " + instruction, 1)
                matches = [(rule, re.fullmatch(rule["rule"], instruction))
                           for rule in grammar["SHFL"]]
                matches = [(rule, match) for rule, match in matches if match]
                self.assertEqual(len(matches), 1, "one grammar must match the full instruction")
                rule, match = matches[0]
                code = GenCode("SHFL", rule, match.groupdict(), line)
                # Internal representation stores the instruction low word in
                # the upper 64 bits; cubin writer emits each word little-endian.
                self.assertEqual(code >> 64, low)
                self.assertEqual(code & ((1 << 64) - 1), high)

    def test_fixture_coverage(self):
        for sm in (70, 75, 80):
            fixture = [instruction for architecture, instruction, _, _ in FIXTURES
                       if architecture == sm]
            self.assertEqual(len(fixture), 24)
            for mode in ("IDX", "BFLY", "UP", "DOWN"):
                self.assertEqual(sum(i.startswith("SHFL." + mode + " ") for i in fixture), 6)
            self.assertTrue(any("0x1f1f" in i for i in fixture))
            self.assertTrue(any(re.search(r", R[0-9]+;$", i) for i in fixture))


if __name__ == "__main__":
    unittest.main()
