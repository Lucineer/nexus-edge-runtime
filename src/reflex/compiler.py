"""Nexus Edge Intent Compiler — natural language intent to bytecode.

Pipeline: Intent parse → IR generation → Safety validation → Bytecode emit
Example: "maintain depth at 2m" → bytecode that reads depth sensor and adjusts thruster
"""
import re, struct
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class Intent:
    raw: str
    action: str = ""       # maintain, navigate, monitor, alert, sequence
    target: str = ""       # depth, heading, speed, temperature, distance
    value: float = 0.0     # setpoint
    unit: str = ""         # m, deg, m/s, C, cm
    condition: str = ""    # if_gt, if_lt, if_eq, always
    threshold: float = 0.0
    output_pin: int = 0    # actuator pin
    input_pin: int = 0     # sensor pin
    priority: int = 0      # 0=normal, 1=high, 2=critical
    loop: bool = True


@dataclass
class IRInstruction:
    opcode: str
    args: List = field(default_factory=list)
    label: str = ""
    comment: str = ""


class IntentParser:
    """Parse natural language intent into structured Intent."""

    TARGETS = {
        "depth": "depth", "altitude": "depth", "height": "depth",
        "heading": "heading", "yaw": "heading", "direction": "heading",
        "speed": "speed", "velocity": "speed", "throttle": "speed",
        "temperature": "temperature", "temp": "temperature",
        "distance": "distance", "range": "distance",
        "battery": "battery", "power": "battery", "voltage": "battery",
        "pressure": "pressure",
    }
    ACTIONS = {
        "maintain": "maintain", "keep": "maintain", "hold": "maintain", "regulate": "maintain",
        "navigate": "navigate", "go": "navigate", "move": "navigate", "travel": "navigate",
        "monitor": "monitor", "watch": "monitor", "track": "monitor", "observe": "monitor",
        "alert": "alert", "warn": "alert", "notify": "alert",
        "surface": "navigate", "dive": "navigate", "ascend": "navigate", "descend": "navigate",
        "stop": "monitor", "halt": "monitor", "emergency": "alert",
    }

    def parse(self, text: str) -> Intent:
        intent = Intent(raw=text)
        text_lower = text.lower().strip()

        # Parse action
        for keyword, action in self.ACTIONS.items():
            if keyword in text_lower:
                intent.action = action
                break

        # Parse target
        for keyword, target in self.TARGETS.items():
            if keyword in text_lower:
                intent.target = target
                break

        # Parse numeric value
        nums = re.findall(r'[\d]+\.?[\d]*', text)
        if nums:
            intent.value = float(nums[0])

        # Parse condition (BEFORE threshold fixup)
        if "above" in text_lower or "exceeds" in text_lower or "greater" in text_lower:
            intent.condition = "if_gt"
        elif "below" in text_lower or "less" in text_lower or "under" in text_lower:
            intent.condition = "if_lt"
        elif "between" in text_lower and len(nums) >= 2:
            intent.condition = "if_range"

        # For conditional intents, the value IS the threshold
        if intent.condition and intent.value > 0:
            intent.threshold = intent.value
            intent.value = 0.0

        # Parse unit
        unit_map = {"m": "m", "meter": "m", "meters": "m",
                   "deg": "deg", "degree": "deg", "degrees": "deg",
                   "m/s": "m/s", "mps": "m/s", "knot": "knot", "knots": "knot",
                   "c": "C", "°c": "C", "°": "deg", "cm": "cm", "mm": "mm",
                   "%%": "%", "%": "%", "v": "V", "volt": "V", "volts": "V",
                   "bar": "bar", "psi": "psi"}
        for keyword, unit in unit_map.items():
            if keyword in text_lower:
                intent.unit = unit
                break

        # Parse priority
        if "critical" in text_lower or "emergency" in text_lower:
            intent.priority = 2
        elif "high" in text_lower or "urgent" in text_lower:
            intent.priority = 1

        return intent


    def emit(self, intent: Intent) -> List[IRInstruction]:
        ir = []
        input_pin = self.PIN_MAP.get(intent.target, 0)

        if intent.action == "maintain":
            # Read sensor, compare with setpoint, adjust output
            ir.append(IRInstruction("READ_PIN", [input_pin], comment=f"Read {intent.target} sensor"))
            ir.append(IRInstruction("PUSH_F32", [intent.value], comment=f"Setpoint: {intent.value}{intent.unit}"))
            ir.append(IRInstruction("SUB_F", [], comment="Error = sensor - setpoint"))
            ir.append(IRInstruction("PUSH_F32", [0.1], comment="Dead band"))
            ir.append(IRInstruction("ABS_F", [], comment="|error|"))
            ir.append(IRInstruction("LT_F", [], comment="In dead band?"))
            ir.append(IRInstruction("JUMP_IF_FALSE", ["adjust"], comment="No → adjust"))
            ir.append(IRInstruction("JUMP", ["loop_end"], comment="Yes → skip"))
            ir.append(IRInstruction("label", [], "adjust"))
            ir.append(IRInstruction("READ_PIN", [input_pin], comment=f"Re-read {intent.target}"))
            ir.append(IRInstruction("PUSH_F32", [intent.value], comment="Setpoint"))
            ir.append(IRInstruction("CLAMP_F", [], comment=f"Clamp output"))
            output_pin = self.PIN_MAP.get("thruster", 10)
            ir.append(IRInstruction("WRITE_PIN", [output_pin], comment=f"Output to pin {output_pin}"))
            ir.append(IRInstruction("label", [], "loop_end"))

        elif intent.action == "navigate":
            ir.append(IRInstruction("PUSH_F32", [intent.value], comment=f"Target: {intent.value}{intent.unit}"))
            output_pin = self.PIN_MAP.get("thruster", 10)
            ir.append(IRInstruction("WRITE_PIN", [output_pin], comment=f"Set output"))
            ir.append(IRInstruction("READ_PIN", [input_pin], comment=f"Check {intent.target}"))

        elif intent.action == "monitor":
            ir.append(IRInstruction("READ_PIN", [input_pin], comment=f"Monitor {intent.target}"))
            if intent.condition == "if_gt" and intent.threshold > 0:
                ir.append(IRInstruction("PUSH_F32", [intent.threshold], comment=f"Threshold: {intent.threshold}"))
                ir.append(IRInstruction("GT_F", [], comment=f"{intent.target} > threshold?"))
                output_pin = self.PIN_MAP.get("led", 13)
                ir.append(IRInstruction("WRITE_PIN", [output_pin], comment="Alert"))
            elif intent.condition == "if_lt" and intent.threshold > 0:
                ir.append(IRInstruction("PUSH_F32", [intent.threshold], comment=f"Threshold: {intent.threshold}"))
                ir.append(IRInstruction("LT_F", [], comment=f"{intent.target} < threshold?"))
                output_pin = self.PIN_MAP.get("buzzer", 14)
                ir.append(IRInstruction("WRITE_PIN", [output_pin], comment="Alert"))

        elif intent.action == "alert":
            ir.append(IRInstruction("PUSH_F32", [1.0], comment="Alert active"))
            output_pin = self.PIN_MAP.get("led", 13)
            ir.append(IRInstruction("WRITE_PIN", [output_pin], comment="LED on"))

        return ir


class IREmitter:
    """Generate intermediate representation from Intent."""

    PIN_MAP = {
        "depth": 1, "heading": 2, "speed": 3, "temperature": 4,
        "distance": 5, "battery": 6, "pressure": 7,
        "thruster": 10, "rudder": 11, "motor": 12, "led": 13,
        "buzzer": 14,
    }

    def emit(self, intent: Intent) -> List[IRInstruction]:
        ir = []
        input_pin = self.PIN_MAP.get(intent.target, 0)

        if intent.action == "maintain":
            ir.append(IRInstruction("READ_PIN", [input_pin], comment=f"Read {intent.target}"))
            ir.append(IRInstruction("PUSH_F32", [intent.value], comment=f"Setpoint: {intent.value}{intent.unit}"))
            ir.append(IRInstruction("SUB_F", [], comment="Error = sensor - setpoint"))
            ir.append(IRInstruction("PUSH_F32", [0.1], comment="Dead band"))
            ir.append(IRInstruction("ABS_F", [], comment="|error|"))
            ir.append(IRInstruction("LT_F", [], comment="In dead band?"))
            ir.append(IRInstruction("JUMP_IF_FALSE", ["adjust"], label="adjust", comment="No -> adjust"))
            ir.append(IRInstruction("JUMP", ["loop_end"], label="loop_end", comment="Yes -> skip"))
            ir.append(IRInstruction("label", [], "adjust"))
            ir.append(IRInstruction("READ_PIN", [input_pin], comment=f"Re-read {intent.target}"))
            ir.append(IRInstruction("PUSH_F32", [intent.value], comment="Setpoint"))
            ir.append(IRInstruction("CLAMP_F", [], comment="Clamp output"))
            output_pin = self.PIN_MAP.get("thruster", 10)
            ir.append(IRInstruction("WRITE_PIN", [output_pin], comment=f"Output to pin {output_pin}"))
            ir.append(IRInstruction("label", [], "loop_end"))

        elif intent.action == "navigate":
            ir.append(IRInstruction("PUSH_F32", [intent.value], comment=f"Target: {intent.value}{intent.unit}"))
            output_pin = self.PIN_MAP.get("thruster", 10)
            ir.append(IRInstruction("WRITE_PIN", [output_pin], comment="Set output"))
            ir.append(IRInstruction("READ_PIN", [input_pin], comment=f"Check {intent.target}"))

        elif intent.action == "monitor":
            ir.append(IRInstruction("READ_PIN", [input_pin], comment=f"Monitor {intent.target}"))
            if intent.condition in ("if_gt", "if_lt") and intent.threshold > 0:
                ir.append(IRInstruction("PUSH_F32", [intent.threshold], comment=f"Threshold: {intent.threshold}"))
                op = "GT_F" if intent.condition == "if_gt" else "LT_F"
                ir.append(IRInstruction(op, [], comment=f"{intent.target} {'>' if intent.condition == 'if_gt' else '<'} threshold?"))
                output_pin = self.PIN_MAP.get("led" if intent.condition == "if_gt" else "buzzer", 13)
                ir.append(IRInstruction("WRITE_PIN", [output_pin], comment="Alert"))

        elif intent.action == "alert":
            ir.append(IRInstruction("PUSH_F32", [1.0], comment="Alert active"))
            output_pin = self.PIN_MAP.get("led", 13)
            ir.append(IRInstruction("WRITE_PIN", [output_pin], comment="LED on"))

        return ir


class BytecodeEmitter:
    """Emit bytecode from IR instructions."""

    OPCODES = {
        "NOP": 0x00, "PUSH_I8": 0x01, "PUSH_I16": 0x02, "PUSH_F32": 0x03,
        "POP": 0x04, "DUP": 0x05, "SWAP": 0x06, "ROT": 0x07,
        "ADD_F": 0x08, "SUB_F": 0x09, "MUL_F": 0x0A, "DIV_F": 0x0B,
        "NEG_F": 0x0C, "ABS_F": 0x0D, "MIN_F": 0x0E, "MAX_F": 0x0F, "CLAMP_F": 0x10,
        "EQ_F": 0x11, "LT_F": 0x12, "GT_F": 0x13, "LTE_F": 0x14, "GTE_F": 0x15,
        "AND_B": 0x16, "OR_B": 0x17, "XOR_B": 0x18, "NOT_B": 0x19,
        "READ_PIN": 0x1A, "WRITE_PIN": 0x1B, "READ_TIMER_MS": 0x1C,
        "JUMP": 0x1D, "JUMP_IF_FALSE": 0x1E, "JUMP_IF_TRUE": 0x1F,
    }

    def emit(self, ir: List[IRInstruction]) -> Tuple[bytes, str]:
        """Emit bytecode and assembly listing."""
        labels: Dict[str, int] = {}
        jumps_to_resolve: List[Tuple[int, str]] = []
        assembly = []
        bytecode = bytearray()

        # First pass: collect labels
        idx = 0
        for instr in ir:
            if instr.opcode == "label":
                lbl = instr.args[0] if instr.args else instr.label
                labels[lbl] = idx
                continue
            idx += 1

        # Second pass: emit
        idx = 0
        for instr in ir:
            if instr.opcode == "label":
                lbl = instr.args[0] if instr.args else instr.label
                assembly.append(f"{lbl}:")
                continue

            opcode_val = self.OPCODES.get(instr.opcode, 0x00)
            arg8 = 0
            arg16 = 0
            imm32 = 0

            if instr.opcode == "PUSH_F32" and instr.args:
                imm32 = struct.unpack('<i', struct.pack('<f', instr.args[0]))[0]
                assembly.append(f"{idx:4d}: PUSH_F32 {instr.args[0]:.2f}")
            elif instr.opcode in ("READ_PIN", "WRITE_PIN") and instr.args:
                arg8 = int(instr.args[0])
                assembly.append(f"{idx:4d}: {instr.opcode} PIN_{arg8}")
            elif instr.opcode in ("JUMP", "JUMP_IF_FALSE", "JUMP_IF_TRUE") and instr.args:
                target = instr.args[0]
                arg16 = labels.get(target, 0)
                jumps_to_resolve.append((idx, target))
                assembly.append(f"{idx:4d}: {instr.opcode} @{target}")
            elif instr.opcode == "PUSH_I8" and instr.args:
                arg8 = int(instr.args[0]) & 0xFF
                assembly.append(f"{idx:4d}: PUSH_I8 {arg8}")
            elif instr.comment:
                assembly.append(f"{idx:4d}: {instr.opcode}  ; {instr.comment}")
            else:
                assembly.append(f"{idx:4d}: {instr.opcode}")

            bytecode.extend(struct.pack('<BBHi', opcode_val, arg8, arg16, imm32 & 0xFFFFFFFF))
            idx += 1

        return bytes(bytecode), "\n".join(assembly)


class IntentCompiler:
    """Full pipeline: text → intent → IR → bytecode."""

    def __init__(self):
        self.parser = IntentParser()
        self.ir_emitter = IREmitter()
        self.bytecode_emitter = BytecodeEmitter()

    def compile(self, text: str) -> Dict:
        intent = self.parser.parse(text)
        ir = self.ir_emitter.emit(intent)
        bytecode, assembly = self.bytecode_emitter.emit(ir)
        return {
            "intent": intent,
            "ir_count": len(ir),
            "bytecode": bytecode,
            "bytecode_size": len(bytecode),
            "assembly": assembly,
        }


def demo():
    print("=== Nexus Edge Intent Compiler ===\n")

    compiler = IntentCompiler()
    intents = [
        "maintain depth at 2.0m",
        "monitor temperature, alert if above 35C",
        "navigate to distance 10m",
        "emergency stop",
        "keep speed at 1.5 m/s",
        "monitor battery, warn if below 15 percent",
        "alert critical depth exceeds 5m",
    ]

    for text in intents:
        print(f"--- '{text}' ---")
        result = compiler.compile(text)
        intent = result["intent"]
        print(f"  Parsed: action={intent.action}, target={intent.target}, "
              f"value={intent.value}{intent.unit}, cond={intent.condition}")
        print(f"  IR: {result['ir_count']} instructions → {result['bytecode_size']} bytes bytecode")
        for line in result["assembly"].split("\n")[:6]:
            print(f"    {line}")
        if len(result["assembly"].split("\n")) > 6:
            print(f"    ...")
        print()


if __name__ == "__main__":
    demo()
