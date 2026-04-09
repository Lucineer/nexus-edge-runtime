"""Nexus Edge Runtime — Bytecode VM for autonomous agent control.

32-opcode stack-based VM with deterministic cycle-count execution.
Designed for microcontroller deployment (ESP32-S3) and edge GPU supervision (Jetson).

Instruction format (8 bytes, little-endian):
    [opcode:u8][arg8:u8][arg16:u16][imm32:i32]

Register file: R0-R15 (GP), R16-R31 (IO-mapped)
Memory: 64KB addressable
Stack: 1024 entries
"""
import struct, math, enum, time, logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any

logger = logging.getLogger(__name__)


class Opcode(enum.IntEnum):
    # Stack (0x00-0x07)
    NOP = 0x00; PUSH_I8 = 0x01; PUSH_I16 = 0x02; PUSH_F32 = 0x03
    POP = 0x04; DUP = 0x05; SWAP = 0x06; ROT = 0x07
    # Arithmetic (0x08-0x10)
    ADD_F = 0x08; SUB_F = 0x09; MUL_F = 0x0A; DIV_F = 0x0B
    NEG_F = 0x0C; ABS_F = 0x0D; MIN_F = 0x0E; MAX_F = 0x0F; CLAMP_F = 0x10
    # Compare (0x11-0x15)
    EQ_F = 0x11; LT_F = 0x12; GT_F = 0x13; LTE_F = 0x14; GTE_F = 0x15
    # Logic (0x16-0x19)
    AND_B = 0x16; OR_B = 0x17; XOR_B = 0x18; NOT_B = 0x19
    # I/O (0x1A-0x1C)
    READ_PIN = 0x1A; WRITE_PIN = 0x1B; READ_TIMER_MS = 0x1C
    # Control (0x1D-0x1F)
    JUMP = 0x1D; JUMP_IF_FALSE = 0x1E; JUMP_IF_TRUE = 0x1F

OPCODE_NAMES = {op.value: op.name for op in Opcode}


class VMError(Exception):
    pass


class HaltError(Exception):
    pass


@dataclass
class VMState:
    pc: int = 0
    stack: List[float] = field(default_factory=list)
    regs: List[float] = field(default_factory=lambda: [0.0] * 32)
    memory: bytearray = field(default_factory=lambda: bytearray(65536))
    cycles: int = 0
    halted: bool = False
    output_log: List[Tuple[str, Any]] = field(default_factory=list)


class BytecodeVM:
    """Deterministic bytecode virtual machine for edge control."""

    MAX_STACK = 1024
    MEM_SIZE = 65536
    INSTR_SIZE = 8

    def __init__(self, io_handler: Optional[Callable] = None):
        self.state = VMState()
        self.io_handler = io_handler or self._default_io
        self.pin_state: Dict[int, float] = {}
        self._start_time = 0.0

    def load(self, bytecode: bytes) -> None:
        n_instr = len(bytecode) // self.INSTR_SIZE
        for i in range(n_instr):
            offset = i * self.INSTR_SIZE
            end = offset + self.INSTR_SIZE
            chunk = bytecode[offset:end]
            if len(chunk) < self.INSTR_SIZE:
                chunk = chunk + b'\x00' * (self.INSTR_SIZE - len(chunk))
            dest = offset
            if dest + len(chunk) <= self.MEM_SIZE:
                self.state.memory[dest:dest + len(chunk)] = chunk

    def reset(self) -> None:
        self.state = VMState()
        self.pin_state = {}
        self._start_time = 0.0

    def _default_io(self, pin: int, value: float, is_write: bool) -> float:
        if is_write:
            self.pin_state[pin] = value
        return self.pin_state.get(pin, 0.0)

    def _push(self, val: float) -> None:
        if len(self.state.stack) >= self.MAX_STACK:
            raise VMError("Stack overflow")
        self.state.stack.append(val)

    def _pop(self) -> float:
        if not self.state.stack:
            raise VMError("Stack underflow")
        return self.state.stack.pop()

    def _read_instr(self, pc: int) -> Tuple[int, int, int, int]:
        if pc + self.INSTR_SIZE > self.MEM_SIZE:
            raise VMError(f"PC out of bounds: {pc}")
        data = self.state.memory[pc:pc + self.INSTR_SIZE]
        opcode = data[0]
        arg8 = data[1]
        arg16 = struct.unpack_from('<H', data, 2)[0]
        imm32 = struct.unpack_from('<i', data, 4)[0]
        return opcode, arg8, arg16, imm32

    def step(self) -> bool:
        if self.state.halted:
            return False
        s = self.state
        op, arg8, arg16, imm32 = self._read_instr(s.pc)
        s.cycles += 1
        s.pc += self.INSTR_SIZE

        if op == Opcode.NOP:
            pass
        elif op == Opcode.PUSH_I8:
            self._push(float(struct.unpack('<b', struct.pack('<B', arg8))[0]))
        elif op == Opcode.PUSH_I16:
            self._push(float(struct.unpack('<h', struct.pack('<H', arg16))[0]))
        elif op == Opcode.PUSH_F32:
            self._push(float(struct.unpack('<f', struct.pack('<i', imm32))[0]))
        elif op == Opcode.POP:
            self._pop()
        elif op == Opcode.DUP:
            v = self._pop()
            self._push(v)
            self._push(v)
        elif op == Opcode.SWAP:
            b = self._pop()
            a = self._pop()
            self._push(b)
            self._push(a)
        elif op == Opcode.ROT:
            if len(s.stack) < 3:
                raise VMError("ROT needs 3 items")
            c = s.stack.pop(-3)
            s.stack.append(c)
        elif op == Opcode.ADD_F:
            b, a = self._pop(), self._pop()
            self._push(a + b)
        elif op == Opcode.SUB_F:
            b, a = self._pop(), self._pop()
            self._push(a - b)
        elif op == Opcode.MUL_F:
            b, a = self._pop(), self._pop()
            self._push(a * b)
        elif op == Opcode.DIV_F:
            b, a = self._pop(), self._pop()
            if b == 0:
                self._push(0.0)
            else:
                self._push(a / b)
        elif op == Opcode.NEG_F:
            self._push(-self._pop())
        elif op == Opcode.ABS_F:
            self._push(abs(self._pop()))
        elif op == Opcode.MIN_F:
            b, a = self._pop(), self._pop()
            self._push(min(a, b))
        elif op == Opcode.MAX_F:
            b, a = self._pop(), self._pop()
            self._push(max(a, b))
        elif op == Opcode.CLAMP_F:
            v = self._pop()
            lo = float(struct.unpack('<f', struct.pack('<i', imm32))[0])
            hi = float(arg8)
            self._push(max(lo, min(hi, v)))
        elif op == Opcode.EQ_F:
            b, a = self._pop(), self._pop()
            self._push(1.0 if a == b else 0.0)
        elif op == Opcode.LT_F:
            b, a = self._pop(), self._pop()
            self._push(1.0 if a < b else 0.0)
        elif op == Opcode.GT_F:
            b, a = self._pop(), self._pop()
            self._push(1.0 if a > b else 0.0)
        elif op == Opcode.LTE_F:
            b, a = self._pop(), self._pop()
            self._push(1.0 if a <= b else 0.0)
        elif op == Opcode.GTE_F:
            b, a = self._pop(), self._pop()
            self._push(1.0 if a >= b else 0.0)
        elif op == Opcode.AND_B:
            b, a = int(self._pop()), int(self._pop())
            self._push(float(a & b))
        elif op == Opcode.OR_B:
            b, a = int(self._pop()), int(self._pop())
            self._push(float(a | b))
        elif op == Opcode.XOR_B:
            b, a = int(self._pop()), int(self._pop())
            self._push(float(a ^ b))
        elif op == Opcode.NOT_B:
            self._push(float(~int(self._pop()) & 0xFF))
        elif op == Opcode.READ_PIN:
            val = self.io_handler(arg8, 0.0, False)
            self._push(val)
            s.regs[16 + arg8 % 16] = val
        elif op == Opcode.WRITE_PIN:
            v = self._pop()
            self.io_handler(arg8, v, True)
            s.output_log.append(("WRITE_PIN", (arg8, v)))
        elif op == Opcode.READ_TIMER_MS:
            elapsed = (time.time() - self._start_time) * 1000 if self._start_time else 0
            self._push(elapsed)
        elif op == Opcode.JUMP:
            s.pc = arg16 * self.INSTR_SIZE
        elif op == Opcode.JUMP_IF_FALSE:
            v = self._pop()
            if v == 0.0:
                s.pc = arg16 * self.INSTR_SIZE
        elif op == Opcode.JUMP_IF_TRUE:
            v = self._pop()
            if v != 0.0:
                s.pc = arg16 * self.INSTR_SIZE
        else:
            raise VMError(f"Unknown opcode: 0x{op:02X} at PC={s.pc - self.INSTR_SIZE}")
        return True

    def run(self, max_cycles: int = 100000) -> VMState:
        self._start_time = time.time()
        while not self.state.halted and self.state.cycles < max_cycles:
            if not self.step():
                break
        return self.state


class Assembler:
    """Assembly mnemonics → bytecode compiler."""

    mnemonics = {op.name: op.value for op in Opcode}

    def assemble(self, source: str) -> bytes:
        bytecode = bytearray()
        labels: Dict[str, int] = {}
        jumps: List[Tuple[int, str]] = []

        lines = [l.split('//')[0].strip() for l in source.strip().split('\n')]
        instr_idx = 0

        # First pass: collect labels
        for line in lines:
            if not line:
                continue
            if line.endswith(':'):
                labels[line[:-1].strip()] = instr_idx
                continue
            parts = line.replace(',', ' ').split()
            if parts and parts[0].upper() in self.mnemonics:
                instr_idx += 1

        # Second pass: encode
        instr_idx = 0
        for line in lines:
            if not line or line.endswith(':'):
                continue
            parts = line.replace(',', ' ').split()
            mnemonic = parts[0].upper()
            if mnemonic not in self.mnemonics:
                raise ValueError(f"Unknown mnemonic: {mnemonic}")

            opcode = self.mnemonics[mnemonic]
            arg8 = 0
            arg16 = 0
            imm32 = 0

            if len(parts) > 1:
                val = parts[1]
                if val in labels:
                    arg16 = labels[val]
                elif opcode in (Opcode.PUSH_I8,):
                    arg8 = int(float(val)) & 0xFF
                    imm32 = struct.unpack('<i', struct.pack('<f', float(val)))[0]
                elif opcode in (Opcode.PUSH_F32,):
                    imm32 = struct.unpack('<i', struct.pack('<f', float(val)))[0]
                elif opcode in (Opcode.PUSH_I16,):
                    arg16 = int(float(val)) & 0xFFFF
                elif opcode in (Opcode.JUMP, Opcode.JUMP_IF_FALSE, Opcode.JUMP_IF_TRUE):
                    if val in labels:
                        arg16 = labels[val]
                    else:
                        arg16 = int(val)
                elif opcode in (Opcode.READ_PIN, Opcode.WRITE_PIN):
                    arg8 = int(val) & 0xFF
                else:
                    try:
                        imm32 = int(float(val))
                    except:
                        if val in labels:
                            arg16 = labels[val]

            bytecode.extend(struct.pack('<BBHi', opcode, arg8, arg16, imm32 & 0xFFFFFFFF))
            instr_idx += 1

        # Fixup jumps
        for offset, label in jumps:
            pass  # already resolved in first pass

        return bytes(bytecode)

    def disassemble(self, bytecode: bytes) -> str:
        lines = []
        for i in range(0, len(bytecode), 8):
            if i + 8 > len(bytecode):
                break
            chunk = bytecode[i:i+8]
            op = chunk[0]
            name = OPCODE_NAMES.get(op, f"UNKNOWN_0x{op:02X}")
            arg8 = chunk[1]
            arg16 = struct.unpack_from('<H', chunk, 2)[0]
            imm32 = struct.unpack_from('<i', chunk, 4)[0]
            addr = i // 8

            if op in (Opcode.PUSH_F32,):
                val = struct.unpack('<f', struct.pack('<i', imm32))[0]
                lines.append(f"{addr:4d}: {name} {val:.2f}")
            elif op in (Opcode.PUSH_I8,):
                lines.append(f"{addr:4d}: {name} {arg8}")
            elif op in (Opcode.READ_PIN, Opcode.WRITE_PIN):
                lines.append(f"{addr:4d}: {name} PIN_{arg8}")
            elif op in (Opcode.JUMP, Opcode.JUMP_IF_FALSE, Opcode.JUMP_IF_TRUE):
                lines.append(f"{addr:4d}: {name} @{arg16}")
            elif op == Opcode.NOP:
                lines.append(f"{addr:4d}: NOP")
            else:
                lines.append(f"{addr:4d}: {name}")

        return "\n".join(lines)


class Validator:
    """Validate bytecode before deployment."""

    def validate(self, bytecode: bytes) -> Tuple[bool, List[str]]:
        errors = []
        if len(bytecode) == 0:
            return False, ["Empty bytecode"]
        if len(bytecode) % 8 != 0:
            errors.append(f"Bytecode length {len(bytecode)} not multiple of 8")
            return False, errors

        seen_ops = set()
        max_pc = 0
        for i in range(0, len(bytecode), 8):
            op = bytecode[i]
            addr = i // 8
            if op not in OPCODE_NAMES:
                errors.append(f"Invalid opcode 0x{op:02X} at instruction {addr}")
            seen_ops.add(op)

        # Check for HALT-like termination (should end with JUMP or reach end)
        last_op = bytecode[-8]
        if last_op not in (Opcode.JUMP, Opcode.JUMP_IF_FALSE, Opcode.JUMP_IF_TRUE,
                          Opcode.NOP, Opcode.WRITE_PIN, Opcode.POP):
            pass  # Not necessarily an error

        return len(errors) == 0, errors


def demo():
    print("=== Nexus Edge Bytecode VM ===\n")

    # Assembler demo
    asm = Assembler()
    program = """
        // Read sensor on pin 5
        PUSH_F32 42.0
        WRITE_PIN 5
        // Loop: read pin 6, write to pin 7
        READ_PIN 6
        WRITE_PIN 7
        PUSH_I8 100
        READ_PIN 8
        ADD_F
        WRITE_PIN 9
        // Conditional
        READ_PIN 6
        PUSH_F32 50.0
        GT_F
        JUMP_IF_FALSE 0
        PUSH_F32 99.9
        WRITE_PIN 10
    """
    bytecode = asm.assemble(program)
    print(f"Assembled: {len(bytecode)} bytes, {len(bytecode)//8} instructions")

    # Disassemble
    print("\n--- Disassembly ---")
    print(asm.disassemble(bytecode))

    # Execute
    print("--- Execution ---")
    vm = BytecodeVM()
    vm.load(bytecode)
    vm.pin_state[6] = 75.0  # simulate sensor reading
    vm.pin_state[8] = 10.0
    state = vm.run(max_cycles=500)
    print(f"  Cycles: {state.cycles}")
    print(f"  Stack depth: {len(state.stack)}")
    print(f"  Pin states: {vm.pin_state}")
    print(f"  Output log: {state.output_log}")

    # Validate
    print("\n--- Validation ---")
    v = Validator()
    ok, errs = v.validate(bytecode)
    print(f"  Valid: {ok}, Errors: {errs}")

    # Safety check: infinite loop detection
    print("\n--- Infinite Loop Detection ---")
    loop_code = """JUMP 0"""
    loop_bc = asm.assemble(loop_code)
    vm2 = BytecodeVM()
    vm2.load(loop_bc)
    state2 = vm2.run(max_cycles=100)
    print(f"  Ran {state2.cycles} cycles (limited to 100)")


if __name__ == "__main__":
    demo()
