"""phase_alignment.py with the SV frame size on the wire (149 B + 20 B).
With 169 B, 16 streams overload the port (rho = 1.04); the last stable BBP
port of the case study has 14 streams (N = 7, rho = 0.91)."""
import runpy
import validate_des as v

v.S_BYTES = 169
v.S = v.S_BYTES * 8 / (v.C * 1e6)
runpy.run_path('phase_alignment.py', init_globals={'KS_OVERRIDE': (8, 12, 14)}, run_name='__main__')
