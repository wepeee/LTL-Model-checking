import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict, deque

BPMN_NS = {"bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL"}

# ------------------------------------------------------------
# BPMN Graph Representation
# ------------------------------------------------------------

@dataclass
class BPMNNode:
    id: str
    name: str
    type: str
    process_id: Optional[str] = None
    pool: Optional[str] = None
    attrs: Dict[str, str] = field(default_factory=dict)

@dataclass
class BPMNEdge:
    id: str
    source: str
    target: str
    type: str  # "sequenceFlow" or "messageFlow"
    attrs: Dict[str, str] = field(default_factory=dict)

class BPMNGraph:
    def __init__(self):
        self.nodes: Dict[str, BPMNNode] = {}
        self.edges: List[BPMNEdge] = []
        self.incoming: Dict[str, List[BPMNEdge]] = {}
        self.outgoing: Dict[str, List[BPMNEdge]] = {}

    def add_node(self, node_id: str, type_: str, name: Optional[str] = None,
                 process_id: Optional[str] = None, pool: Optional[str] = None,
                 attrs: Optional[Dict[str, str]] = None):
        if node_id in self.nodes:
            # update attrs kalau sudah ada
            node = self.nodes[node_id]
            if name and not node.name:
                node.name = name
            if process_id:
                node.process_id = process_id
            if pool:
                node.pool = pool
            if attrs:
                node.attrs.update(attrs)
            return

        if name is None:
            name = node_id
        if attrs is None:
            attrs = {}
        node = BPMNNode(
            id=node_id,
            name=name,
            type=type_,
            process_id=process_id,
            pool=pool,
            attrs=attrs
        )
        self.nodes[node_id] = node
        self.incoming.setdefault(node_id, [])
        self.outgoing.setdefault(node_id, [])

    def add_edge(self, edge_id: str, source: str, target: str,
                 type_: str, attrs: Optional[Dict[str, str]] = None):
        if attrs is None:
            attrs = {}
        edge = BPMNEdge(
            id=edge_id,
            source=source,
            target=target,
            type=type_,
            attrs=attrs
        )
        self.edges.append(edge)
        self.outgoing.setdefault(source, []).append(edge)
        self.incoming.setdefault(target, []).append(edge)

    def get_node(self, node_id: str) -> BPMNNode:
        return self.nodes[node_id]

# ------------------------------------------------------------
# BPMN XML Parsing (minimal)
# ------------------------------------------------------------

def parse_bpmn_xml(path: str) -> BPMNGraph:
    tree = ET.parse(path)
    root = tree.getroot()
    g = BPMNGraph()

    # map process id -> pool (participant) name
    process_to_pool: Dict[str, str] = {}

    # collaborations: participants and message flows
    for collab in root.findall("bpmn:collaboration", BPMN_NS):
        for participant in collab.findall("bpmn:participant", BPMN_NS):
            pid = participant.get("processRef")
            pool_name = participant.get("name") or participant.get("id")
            if pid:
                process_to_pool[pid] = pool_name

    # processes and their flow nodes
    flow_node_tags = [
        "task",
        "userTask",
        "serviceTask",
        "receiveTask",
        "sendTask",
        "scriptTask",
        "subProcess",
        "startEvent",
        "endEvent",
        "intermediateCatchEvent",
        "intermediateThrowEvent",
        "exclusiveGateway",
        "parallelGateway",
        "inclusiveGateway",
        "eventBasedGateway",
        "complexGateway",
    ]

    for proc in root.findall("bpmn:process", BPMN_NS):
        pid = proc.get("id")
        pool_name = process_to_pool.get(pid)
        for tag in flow_node_tags:
            for el in proc.findall(f"bpmn:{tag}", BPMN_NS):
                node_id = el.get("id")
                name = el.get("name") or node_id
                g.add_node(
                    node_id=node_id,
                    type_=tag,
                    name=name,
                    process_id=pid,
                    pool=pool_name,
                )

        # sequence flows
        for sf in proc.findall("bpmn:sequenceFlow", BPMN_NS):
            eid = sf.get("id")
            src = sf.get("sourceRef")
            tgt = sf.get("targetRef")
            g.add_edge(
                edge_id=eid,
                source=src,
                target=tgt,
                type_="sequenceFlow",
                attrs={"name": sf.get("name") or eid},
            )

    # message flows (collaboration level)
    for collab in root.findall("bpmn:collaboration", BPMN_NS):
        for mf in collab.findall("bpmn:messageFlow", BPMN_NS):
            mid = mf.get("id")
            src = mf.get("sourceRef")
            tgt = mf.get("targetRef")
            g.add_edge(
                edge_id=mid,
                source=src,
                target=tgt,
                type_="messageFlow",
                attrs={"name": mf.get("name") or mid},
            )

    return g

# ------------------------------------------------------------
# Transition System
# ------------------------------------------------------------

class TransitionSystem:
    def __init__(self):
        self.states: Set[str] = set()
        self.initial_states: Set[str] = set()
        self.transitions: Dict[str, Set[str]] = {}
        self.labeling: Dict[str, Set[str]] = {}

    def add_state(self, state: str, labels: Optional[Set[str]] = None, initial: bool = False):
        self.states.add(state)
        self.transitions.setdefault(state, set())
        self.labeling.setdefault(state, set())
        if labels:
            self.labeling[state].update(labels)
        if initial:
            self.initial_states.add(state)

    def add_transition(self, src: str, dst: str):
        self.transitions.setdefault(src, set()).add(dst)

def bpmn_to_transition_system(g: BPMNGraph) -> TransitionSystem:
    ts = TransitionSystem()

    # create states
    for node_id, node in g.nodes.items():
        labels: Set[str] = set()
        labels.add(f"node_{node_id}")
        labels.add(f"type_{node.type}")
        if node.pool:
            labels.add(f"pool_{node.pool}")
        if node.type == "startEvent":
            labels.add("START")
        if node.type == "endEvent":
            labels.add("END")
        ts.add_state(node_id, labels=labels, initial=(node.type == "startEvent"))

    # transitions: sequence + message flows
    for e in g.edges:
        if e.type in ("sequenceFlow", "messageFlow"):
            if e.source in ts.states and e.target in ts.states:
                ts.add_transition(e.source, e.target)

    return ts

# ------------------------------------------------------------
# LTL AST & Parser
# ------------------------------------------------------------

class Formula:
    pass

@dataclass
class Prop(Formula):
    name: str

@dataclass
class UnaryOp(Formula):
    op: str  # 'G','F','X','NOT'
    sub: Formula

@dataclass
class BinaryOp(Formula):
    op: str  # 'AND','OR','IMPLIES','U'
    left: Formula
    right: Formula

class Token:
    def __init__(self, type_: str, value: str):
        self.type = type_
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, {self.value})"

def tokenize(formula_str: str) -> List[Token]:
    s = formula_str.strip()
    tokens: List[Token] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c.isalpha() or c == "_":
            # identifier or temporal operator
            j = i + 1
            while j < len(s) and (s[j].isalnum() or s[j] in "_"):
                j += 1
            text = s[i:j]
            if text in ("G", "F", "X", "U"):
                tokens.append(Token("OP", text))
            else:
                tokens.append(Token("IDENT", text))
            i = j
        elif c.isdigit():
            j = i + 1
            while j < len(s) and (s[j].isalnum() or s[j] in "_"):
                j += 1
            text = s[i:j]
            tokens.append(Token("IDENT", text))
            i = j
        elif c == "(":
            tokens.append(Token("LPAREN", c))
            i += 1
        elif c == ")":
            tokens.append(Token("RPAREN", c))
            i += 1
        elif c == "!":
            tokens.append(Token("OP", "!"))
            i += 1
        elif c in {"∧", "∨", "¬"}:
            if c == "∧":
                tokens.append(Token("OP", "AND"))
            elif c == "∨":
                tokens.append(Token("OP", "OR"))
            elif c == "¬":
                tokens.append(Token("OP", "!"))
            i += 1
        elif c == "&":
            if i + 1 < len(s) and s[i+1] == "&":
                tokens.append(Token("OP", "AND"))
                i += 2
            else:
                tokens.append(Token("OP", "AND"))
                i += 1
        elif c == "|":
            if i + 1 < len(s) and s[i+1] == "|":
                tokens.append(Token("OP", "OR"))
                i += 2
            else:
                tokens.append(Token("OP", "OR"))
                i += 1
        elif c == "-":
            if i + 1 < len(s) and s[i+1] == ">":
                tokens.append(Token("OP", "IMPLIES"))
                i += 2
            else:
                raise ValueError(f"Unexpected '-' at position {i}")
        else:
            raise ValueError(f"Unexpected character '{c}' in formula: {formula_str}")
    return tokens

class LTLParser:
    # precedence: IMPLIES -> OR -> AND -> U -> UNARY
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current(self) -> Optional[Token]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def eat(self, expected_type: Optional[str] = None, expected_value: Optional[str] = None) -> Token:
        tok = self.current()
        if tok is None:
            raise ValueError("Unexpected end of input")
        if expected_type and tok.type != expected_type:
            raise ValueError(f"Expected token type {expected_type} but got {tok.type}")
        if expected_value and tok.value != expected_value:
            raise ValueError(f"Expected token value {expected_value} but got {tok.value}")
        self.pos += 1
        return tok

    def parse(self) -> Formula:
        node = self.parse_implication()
        if self.current() is not None:
            raise ValueError(f"Unexpected token at end: {self.current()}")
        return node

    def parse_implication(self) -> Formula:
        left = self.parse_or()
        tok = self.current()
        if tok is not None and tok.type == "OP" and tok.value == "IMPLIES":
            self.eat("OP", "IMPLIES")
            right = self.parse_implication()
            return BinaryOp(op="IMPLIES", left=left, right=right)
        return left

    def parse_or(self) -> Formula:
        node = self.parse_and()
        while True:
            tok = self.current()
            if tok is not None and tok.type == "OP" and tok.value == "OR":
                self.eat("OP", "OR")
                right = self.parse_and()
                node = BinaryOp(op="OR", left=node, right=right)
            else:
                break
        return node

    def parse_and(self) -> Formula:
        node = self.parse_until()
        while True:
            tok = self.current()
            if tok is not None and tok.type == "OP" and tok.value == "AND":
                self.eat("OP", "AND")
                right = self.parse_until()
                node = BinaryOp(op="AND", left=node, right=right)
            else:
                break
        return node

    def parse_until(self) -> Formula:
        node = self.parse_unary()
        tok = self.current()
        if tok is not None and tok.type == "OP" and tok.value == "U":
            self.eat("OP", "U")
            right = self.parse_until()
            return BinaryOp(op="U", left=node, right=right)
        return node

    def parse_unary(self) -> Formula:
        tok = self.current()
        if tok is None:
            raise ValueError("Unexpected end of input in unary")
        if tok.type == "OP" and tok.value in ("G", "F", "X", "!"):
            self.eat("OP")
            sub = self.parse_unary()
            op = tok.value
            if op == "!":
                op = "NOT"
            return UnaryOp(op=op, sub=sub)
        elif tok.type == "LPAREN":
            self.eat("LPAREN")
            node = self.parse_implication()
            self.eat("RPAREN")
            return node
        elif tok.type == "IDENT":
            self.eat("IDENT")
            return Prop(name=tok.value)
        else:
            raise ValueError(f"Unexpected token in unary: {tok}")

def parse_ltl(formula_str: str) -> Formula:
    tokens = tokenize(formula_str)
    parser = LTLParser(tokens)
    return parser.parse()

# ------------------------------------------------------------
# LTL Evaluation on Finite Paths (LTLf semantics)
# ------------------------------------------------------------

def eval_formula(formula: Formula, path: List[str],
                 labeling: Dict[str, Set[str]], pos: int = 0) -> bool:
    if isinstance(formula, Prop):
        state = path[pos]
        return formula.name in labeling.get(state, set())
    if isinstance(formula, UnaryOp):
        op = formula.op
        if op == "NOT":
            return not eval_formula(formula.sub, path, labeling, pos)
        if op == "X":
            if pos + 1 >= len(path):
                return False
            return eval_formula(formula.sub, path, labeling, pos + 1)
        if op == "F":
            for i in range(pos, len(path)):
                if eval_formula(formula.sub, path, labeling, i):
                    return True
            return False
        if op == "G":
            for i in range(pos, len(path)):
                if not eval_formula(formula.sub, path, labeling, i):
                    return False
            return True
        raise ValueError(f"Unknown unary operator: {op}")
    if isinstance(formula, BinaryOp):
        op = formula.op
        if op == "AND":
            return eval_formula(formula.left, path, labeling, pos) and \
                   eval_formula(formula.right, path, labeling, pos)
        if op == "OR":
            return eval_formula(formula.left, path, labeling, pos) or \
                   eval_formula(formula.right, path, labeling, pos)
        if op == "IMPLIES":
            left_val = eval_formula(formula.left, path, labeling, pos)
            right_val = eval_formula(formula.right, path, labeling, pos)
            return (not left_val) or right_val
        if op == "U":
            for j in range(pos, len(path)):
                if eval_formula(formula.right, path, labeling, j):
                    for k in range(pos, j):
                        if not eval_formula(formula.left, path, labeling, k):
                            return False
                    return True
            return False
        raise ValueError(f"Unknown binary operator: {op}")
    raise ValueError(f"Unknown formula type: {formula}")

# ------------------------------------------------------------
# LTL Model Checking via DFS over all finite paths
# ------------------------------------------------------------

@dataclass
class LTLPropertyResult:
    name: str
    formula_str: str
    formula: Formula
    passed: bool
    counterexample: Optional[List[str]] = None
    error: Optional[str] = None

def check_ltl_property(ts: TransitionSystem, formula: Formula,
                       max_depth: int = 50) -> Tuple[bool, Optional[List[str]]]:
    def dfs(path: List[str]) -> Optional[List[str]]:
        current = path[-1]
        successors = list(ts.transitions.get(current, []))
        is_terminal = (not successors) or (len(path) >= max_depth)
        if is_terminal:
            if not eval_formula(formula, path, ts.labeling, 0):
                return list(path)
            return None
        for nxt in successors:
            path.append(nxt)
            ce = dfs(path)
            if ce is not None:
                return ce
            path.pop()
        return None

    initials = ts.initial_states or ts.states
    for init in initials:
        ce = dfs([init])
        if ce is not None:
            return False, ce
    return True, None

# ------------------------------------------------------------
# Structural BPMN Rule Checking (contoh)
# ------------------------------------------------------------

@dataclass
class RuleViolation:
    rule_id: str
    category: str  # "bpmn_spec" | "style" | "ltl"
    description: str
    node_ids: List[str]
    details: str

def check_bpmn_rules(g: BPMNGraph) -> List[RuleViolation]:
    violations: List[RuleViolation] = []

    # BPMN 0105. Start event tidak boleh punya incoming sequence flow.
    for node_id, node in g.nodes.items():
        if node.type == "startEvent":
            incoming_sf = [
                e for e in g.incoming.get(node_id, []) if e.type == "sequenceFlow"
            ]
            if incoming_sf:
                violations.append(
                    RuleViolation(
                        rule_id="BPMN_0105",
                        category="bpmn_spec",
                        description="Start event cannot have incoming sequence flow",
                        node_ids=[node_id],
                        details=f"Start event {node_id} has {len(incoming_sf)} incoming sequence flows.",
                    )
                )

    # BPMN 0124. End event tidak boleh punya outgoing sequence flow.
    for node_id, node in g.nodes.items():
        if node.type == "endEvent":
            outgoing_sf = [
                e for e in g.outgoing.get(node_id, []) if e.type == "sequenceFlow"
            ]
            if outgoing_sf:
                violations.append(
                    RuleViolation(
                        rule_id="BPMN_0124",
                        category="bpmn_spec",
                        description="End event cannot have outgoing sequence flow",
                        node_ids=[node_id],
                        details=f"End event {node_id} has {len(outgoing_sf)} outgoing sequence flows.",
                    )
                )

    # BPMN 0132/0133. Gateway tidak boleh punya message flow.
    for node_id, node in g.nodes.items():
        if "Gateway" in node.type:
            incoming_mf = [
                e for e in g.incoming.get(node_id, []) if e.type == "messageFlow"
            ]
            outgoing_mf = [
                e for e in g.outgoing.get(node_id, []) if e.type == "messageFlow"
            ]
            if incoming_mf or outgoing_mf:
                violations.append(
                    RuleViolation(
                        rule_id="BPMN_0132_0133",
                        category="bpmn_spec",
                        description="Gateway cannot have incoming or outgoing message flow",
                        node_ids=[node_id],
                        details=f"Gateway {node_id} has {len(incoming_mf)} incoming and {len(outgoing_mf)} outgoing message flows.",
                    )
                )

    # Style 0103. Activities should be labeled.
    for node_id, node in g.nodes.items():
        if "Task" in node.type or node.type == "subProcess":
            if not node.name or node.name.strip() == "":
                violations.append(
                    RuleViolation(
                        rule_id="STYLE_0103",
                        category="style",
                        description="Activities should be labeled",
                        node_ids=[node_id],
                        details=f"Activity {node_id} has empty name.",
                    )
                )
    return violations

# ------------------------------------------------------------
# Property file parsing: "name = formula"
# ------------------------------------------------------------

def load_ltl_properties(path: str) -> List[Tuple[str, str]]:
    props: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"Invalid property line (missing '='): {line}")
            name, formula = line.split("=", 1)
            name = name.strip()
            formula = formula.strip()
            props.append((name, formula))
    return props

# ------------------------------------------------------------
# Neo4j Cypher Export
# ------------------------------------------------------------

def export_violations_to_cypher(violations: List[RuleViolation],
                                ltl_results: List[LTLPropertyResult],
                                path: str):
    lines: List[str] = []

    # Structural / style violations
    for v in violations:
        for nid in v.node_ids:
            rule_id = v.rule_id
            reason = f"{rule_id}: {v.description}"
            cyph = (
                f"MATCH (n:BPMNNode {{id: {json.dumps(nid)}}})\n"
                f"SET n.violated = true,\n"
                f"    n.violated_rules = coalesce(n.violated_rules, []) + {json.dumps(rule_id)},\n"
                f"    n.violation_reasons = coalesce(n.violation_reasons, []) + {json.dumps(reason)};"
            )
            lines.append(cyph)

    # LTL violations: semua state di counterexample path
    for r in ltl_results:
        if not r.passed and r.counterexample:
            rule_id = f"LTL_{r.name}"
            reason = f"{rule_id}: {r.formula_str}"
            for state in r.counterexample:
                cyph = (
                    f"MATCH (n:BPMNNode {{id: {json.dumps(state)}}})\n"
                    f"SET n.violated = true,\n"
                    f"    n.violated_rules = coalesce(n.violated_rules, []) + {json.dumps(rule_id)},\n"
                    f"    n.violation_reasons = coalesce(n.violation_reasons, []) + {json.dumps(reason)};"
                )
                lines.append(cyph)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines))


# ------------------------------------------------------------
# Pretty printing report
# ------------------------------------------------------------

def print_report(model_name: str,
                 ts: TransitionSystem,
                 violations: List[RuleViolation],
                 ltl_results: List[LTLPropertyResult]):
    title = f"{model_name} LTL VERIFICATION"
    border = "═" * (len(title) + 2)
    print(f"╔{border}╗")
    print(f"║ {title} ║")
    print(f"╚{border}╝")
    print()

    print("[1/5] Loading BPMN and building graph...")
    print(f"      ✓ Loaded {len(ts.states)} states, "
          f"{sum(len(v) for v in ts.transitions.values())} transitions")
    print()
    print("[2/5] Building Kripke structure...")
    print("      ✓ Using finite-trace semantics (LTLf)")
    print()
    print("[3/5] Initializing LTL model checker...")
    print("      ✓ Ready")
    print()
    print("[4/5] Loading LTL properties...")
    print(f"      ✓ {len(ltl_results)} properties to verify")
    print()
    print("[5/5] Verifying properties...")
    print()

    passed = 0
    failed = 0
    errors = 0
    for idx, res in enumerate(ltl_results, start=1):
        print(f"[{idx}/{len(ltl_results)}] Checking: {res.name}")
        print(f"    Formula: {res.formula_str}")
        if res.error:
            print("    ! ERROR:", res.error)
            errors += 1
        elif res.passed:
            print("    ✓ PASSED")
            passed += 1
        else:
            print("    ✗ FAILED")
            failed += 1
            if res.counterexample:
                print("    Counterexample path:", " -> ".join(res.counterexample))
        print()

    print("══════════════════════════════════════════════════════════════════════")
    print("VERIFICATION REPORT")
    print("══════════════════════════════════════════════════════════════════════")
    print()
    print(f"Total Properties: {len(ltl_results)}")
    print(f"  ✓ Passed: {passed}")
    print(f"  ✗ Failed: {failed}")
    print(f"  ⚠ Errors: {errors}")
    print()
    if failed > 0:
        print("══════════════════════════════════════════════════════════════════════")
        print("🚨 FAILED PROPERTIES")
        print("══════════════════════════════════════════════════════════════════════")
        print()
        for res in ltl_results:
            if not res.passed and not res.error:
                print(f"❌ {res.name}")
                print(f"   Formula: {res.formula_str}")
                if res.counterexample:
                    print("   Counterexample path:")
                    for state in res.counterexample:
                        print(f"     → {state}")
                print()
    if violations:
        print("══════════════════════════════════════════════════════════════════════")
        print("STRUCTURAL / STYLE VIOLATIONS")
        print("══════════════════════════════════════════════════════════════════════")
        print()
        for v in violations:
            print(f"❌ {v.rule_id} [{v.category}] - {v.description}")
            print(f"   Nodes: {', '.join(v.node_ids)}")
            print(f"   Details: {v.details}")
            print()

from collections import deque, defaultdict

def enrich_labels(g: BPMNGraph, ts: TransitionSystem):
    """
    Label semua AP yang dipakai 4 formula LTL:
      - exLoop, Final
      - exXORsplit, exXORjoin
      - path1, path2, ANDjoin_waiting
      - exGateway1, exGateway2

    Catatan:
      - TIDAK pernah memberi label 'deadlock' ke state mana pun.
      - source_deadlock (2.10) hanya FAIL kalau XOR-split benar-benar
        tidak punya join struktural; XOR yang ditutup AND masuknya ke
        improper_structuring (2.11), bukan source.
    """

    # ----------------------------------------------------------
    # 0. Siapkan labeling kosong dan graph sequenceFlow
    # ----------------------------------------------------------
    for s in ts.states:
        ts.labeling.setdefault(s, set())

    seq_succ: dict[str, list[str]] = {
        nid: [e.target for e in g.outgoing.get(nid, []) if e.type == "sequenceFlow"]
        for nid in g.nodes
    }
    seq_pred: dict[str, list[str]] = {
        nid: [e.source for e in g.incoming.get(nid, []) if e.type == "sequenceFlow"]
        for nid in g.nodes
    }

    # ----------------------------------------------------------
    # 1. Final (end event)
    # ----------------------------------------------------------
    for nid, node in g.nodes.items():
        t = (node.type or "").lower()
        if "endevent" in t:
            ts.labeling[nid].add("Final")

    # ----------------------------------------------------------
    # 2. exLoop  (node yang berada di dalam loop) via SCC di TS
    # ----------------------------------------------------------
    index = 0
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    stack: list[str] = []
    onstack: set[str] = set()
    sccs: list[list[str]] = []

    def strongconnect(v: str):
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)

        for w in ts.transitions.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in onstack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            comp: list[str] = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in ts.states:
        if v not in indices:
            strongconnect(v)

    loop_nodes: set[str] = set()
    for comp in sccs:
        if len(comp) > 1:
            loop_nodes.update(comp)
        elif len(comp) == 1:
            v = comp[0]
            if v in ts.transitions.get(v, []):  # self-loop
                loop_nodes.add(v)

    for nid in loop_nodes:
        ts.labeling.setdefault(nid, set()).add("exLoop")

    # ----------------------------------------------------------
    # 3. ANDjoin_waiting = parallelGateway dengan >1 incoming
    # ----------------------------------------------------------
    for nid, node in g.nodes.items():
        ins = seq_pred.get(nid, [])
        if node.type == "parallelGateway" and len(ins) > 1:
            ts.labeling[nid].add("ANDjoin_waiting")

    # helper: cek apakah node kandidat join (gateway dengan >1 incoming)
    def is_join_candidate(node_id: str) -> bool:
        node = g.nodes[node_id]
        ins = seq_pred.get(node_id, [])
        return node.type in ("exclusiveGateway", "parallelGateway") and len(ins) > 1

    # ----------------------------------------------------------
    # 4. XOR-split: exXORsplit, path1, path2 + simpan cabang split
    # ----------------------------------------------------------
    branches: dict[str, tuple[set[str], set[str]]] = {}

    def collect_branch_nodes(start_id: str) -> set[str]:
        """
        Semua node di cabang dari suatu output XOR.
        BFS lewat sequenceFlow, tapi STOP di join
        (tidak lewat gateway dengan >1 incoming).
        """
        result: set[str] = set()
        dq = deque([start_id])
        while dq:
            v = dq.popleft()
            if v in result:
                continue
            result.add(v)
            if is_join_candidate(v):
                # jangan lewat join; cukup sampai di sini
                continue
            for w in seq_succ.get(v, []):
                dq.append(w)
        return result

    for nid, node in g.nodes.items():
        if node.type == "exclusiveGateway":
            outs = seq_succ.get(nid, [])
            if len(outs) > 1:
                # gateway ini XOR-split
                ts.labeling[nid].add("exXORsplit")

                # dua cabang pertama
                b1 = collect_branch_nodes(outs[0])
                b2 = collect_branch_nodes(outs[1])
                branches[nid] = (b1, b2)

                # label path1/path2 + propagate exXORsplit ke node di cabang
                for v in b1:
                    ts.labeling.setdefault(v, set()).add("path1")
                    ts.labeling[v].add("exXORsplit")
                for v in b2:
                    ts.labeling.setdefault(v, set()).add("path2")
                    ts.labeling[v].add("exXORsplit")

    # ----------------------------------------------------------
    # 5. exXORjoin = gateway (XOR / AND) yang menutup cabang XOR
    #    -> punya predecessor dari cabang1 dan cabang2 split tsb
    #    -> AND-join yang gabung 2 cabang XOR akan jadi exXORjoin,
    #       sehingga TIDAK dihitung sebagai source deadlock.
    # ----------------------------------------------------------
    for split_id, (b1, b2) in branches.items():
        for nid, node in g.nodes.items():
            if node.type not in ("exclusiveGateway", "parallelGateway"):
                continue
            preds = set(seq_pred.get(nid, []))
            if len(preds) < 2:
                continue
            if preds & b1 and preds & b2:
                ts.labeling.setdefault(nid, set()).add("exXORjoin")

    # ----------------------------------------------------------
    # 6. exGateway1 & exGateway2 untuk conditional_livelock
    #    -> ambil dua gateway pertama yang berada di dalam loop (kalau ada)
    # ----------------------------------------------------------
    gateways_in_loops = [
        nid for nid in loop_nodes
        if "gateway" in (g.nodes[nid].type or "").lower()
    ]
    if len(gateways_in_loops) >= 2:
        ts.labeling[gateways_in_loops[0]].add("exGateway1")
        ts.labeling[gateways_in_loops[1]].add("exGateway2")


def mark_conditional_livelock_gateways(g: BPMNGraph, ts: TransitionSystem):
    """
    Tandai gateway yang berada dalam 'conditional livelock cluster' ala Kurnia.

    Definisi cluster:
      - SCC (cycle) di graf sequenceFlow,
      - reachable dari start,
      - tidak ada path dari node SCC ke end event,
      - mengandung >= 2 gateway,
      - ada minimal satu edge langsung gateway -> gateway
        dengan tipe gateway yang berbeda (mis. XOR -> AND).

    Semua gateway di SCC tersebut diberi label:
      - exGateway1
      - exGateway2

    Sehingga formula Kurnia:
      G((exGateway1 ∧ exGateway2) → F(¬F Final))
    bernilai true jika dan hanya jika cluster ini benar-benar ada
    (ingat: di state-state cluster, F Final sudah false selamanya).
    """

    # --------------------- build adjacency (sequenceFlow) ---------------------
    succ: dict[str, list[str]] = {
        nid: [e.target for e in g.outgoing.get(nid, []) if e.type == "sequenceFlow"]
        for nid in g.nodes
    }
    pred: dict[str, list[str]] = {
        nid: [e.source for e in g.incoming.get(nid, []) if e.type == "sequenceFlow"]
        for nid in g.nodes
    }

    # --------------------- start & end nodes ---------------------
    start_nodes = [
        nid for nid, n in g.nodes.items()
        if (n.type or "").lower().startswith("startevent")
    ]
    end_nodes = [
        nid for nid, n in g.nodes.items()
        if "endevent" in (n.type or "").lower()
    ]

    # --------------------- reachable from start ---------------------
    reachable_from_start: set[str] = set()
    dq = deque(start_nodes)
    while dq:
        v = dq.popleft()
        if v in reachable_from_start:
            continue
        reachable_from_start.add(v)
        for w in succ.get(v, []):
            dq.append(w)

    # --------------------- can reach end (reverse BFS) ---------------------
    can_reach_end: set[str] = set()
    dq = deque(end_nodes)
    while dq:
        v = dq.popleft()
        if v in can_reach_end:
            continue
        can_reach_end.add(v)
        for w in pred.get(v, []):
            dq.append(w)

    # --------------------- SCC (Tarjan) di graf BPMN ---------------------
    index = 0
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    stack: list[str] = []
    onstack: set[str] = set()
    sccs: list[list[str]] = []

    def strongconnect(v: str):
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)

        for w in succ.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in onstack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            comp: list[str] = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in g.nodes.keys():
        if v not in indices:
            strongconnect(v)

    # --------------------- bersihkan label lama exGateway1/2 ---------------------
    for labels in ts.labeling.values():
        labels.discard("exGateway1")
        labels.discard("exGateway2")

    # --------------------- pilih SCC yang termasuk conditional livelock ---------------------
    for comp in sccs:
        comp_set = set(comp)

        # harus cycle (lebih dari 1 node, atau self-loop)
        is_cycle = (
            len(comp_set) > 1
            or (len(comp_set) == 1 and comp[0] in succ.get(comp[0], []))
        )
        if not is_cycle:
            continue

        # reachable dari start?
        if not (comp_set & reachable_from_start):
            continue

        # kalau salah satu node bisa reach end -> bukan livelock
        if comp_set & can_reach_end:
            continue

        # kumpulkan gateway dalam SCC ini
        gateways = [
            nid for nid in comp_set
            if "gateway" in (g.nodes[nid].type or "").lower()
        ]
        if len(gateways) < 2:
            continue

        # cek ada edge langsung gateway -> gateway dengan tipe beda
        mismatch_edge_found = False
        for u in gateways:
            for v in succ.get(u, []):
                if v in gateways:
                    t1 = (g.nodes[u].type or "").lower()
                    t2 = (g.nodes[v].type or "").lower()
                    if t1 != t2:  # mismatch tipe, mis. exclusiveGateway vs parallelGateway
                        mismatch_edge_found = True
                        break
            if mismatch_edge_found:
                break

        if not mismatch_edge_found:
            # cluster gateway semua XOR atau semua AND dsb, bukan conditional livelock Kurnia
            continue

        # --------------------- SCC ini adalah conditional livelock cluster ---------------------
        # tandai semua gateway di dalamnya
        for nid in gateways:
            ts.labeling.setdefault(nid, set()).add("exGateway1")
            ts.labeling[nid].add("exGateway2")



def detect_livelock_structural(g: BPMNGraph):
    """
    Deteksi livelock secara struktural (tanpa LTL):
      - Bangun graf sequenceFlow.
      - Cari SCC (cycle) yang reachable dari start dan
        tidak punya path ke end event.

    Return:
      - livelock_components: list of sets of node ids yang masuk livelock.
    """

    # ---------- build adjacency ----------
    succ = {
        nid: [e.target for e in g.outgoing.get(nid, []) if e.type == "sequenceFlow"]
        for nid in g.nodes
    }
    pred = {
        nid: [e.source for e in g.incoming.get(nid, []) if e.type == "sequenceFlow"]
        for nid in g.nodes
    }

    # ---------- start & end nodes ----------
    start_nodes = [nid for nid, n in g.nodes.items()
                   if (n.type or "").lower().startswith("startevent")]
    end_nodes = [nid for nid, n in g.nodes.items()
                 if "endevent" in (n.type or "").lower()]

    # ---------- 1. reachable from start ----------
    reachable_from_start = set()
    dq = deque(start_nodes)
    while dq:
        v = dq.popleft()
        if v in reachable_from_start:
            continue
        reachable_from_start.add(v)
        for w in succ.get(v, []):
            dq.append(w)

    # ---------- 2. can_reach_end (reverse BFS) ----------
    can_reach_end = set()
    dq = deque(end_nodes)
    while dq:
        v = dq.popleft()
        if v in can_reach_end:
            continue
        can_reach_end.add(v)
        for w in pred.get(v, []):
            dq.append(w)

    # ---------- 3. SCC (Tarjan) di graf BPMN ----------
    index = 0
    indices = {}
    lowlink = {}
    stack = []
    onstack = set()
    sccs = []

    def strongconnect(v: str):
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)

        for w in succ.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in onstack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            comp = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in g.nodes.keys():
        if v not in indices:
            strongconnect(v)

    # ---------- 4. livelock SCC selection ----------
    livelock_components = []

    for comp in sccs:
        comp_set = set(comp)
        # cycle? (lebih dari 1 node atau self-loop)
        is_cycle = (
            len(comp_set) > 1 or
            (len(comp_set) == 1 and comp[0] in succ.get(comp[0], []))
        )
        if not is_cycle:
            continue

        # reachable dari start?
        if not (comp_set & reachable_from_start):
            continue

        # ada path ke end? kalau ADA, bukan livelock
        if any(v in can_reach_end for v in comp_set):
            continue

        # kalau sampai sini: SCC ini livelock
        livelock_components.append(comp_set)

    return livelock_components



def mark_improper_structuring_deadlocks(g: BPMNGraph, ts: TransitionSystem):
    """
    Deteksi source deadlock:
    XOR-split yang cabang-cabangnya ketemu di AND-join (parallelGateway) dengan >1 incoming.
    Hasil: node AND-join yang bermasalah dilabeli 'source_deadlock' dan 'deadlock'.
    """

    # adjacency utk sequenceFlow
    succ: Dict[str, List[str]] = {
        nid: [
            e.target
            for e in g.outgoing.get(nid, [])
            if e.type in ("sequenceFlow", "messageFlow")
        ]
        for nid in g.nodes
    }
    pred: Dict[str, List[str]] = {
        nid: [e.source for e in g.incoming.get(nid, []) if e.type == "sequenceFlow"]
        for nid in g.nodes
    }

    # 1. cari XOR-split
    xor_splits: List[str] = []
    for nid, node in g.nodes.items():
        if node.type == "exclusiveGateway":
            outs = [e for e in g.outgoing.get(nid, []) if e.type == "sequenceFlow"]
            if len(outs) > 1:
                xor_splits.append(nid)

    if not xor_splits:
        return  # gak ada pola source-deadlock

    # 2. branch_labels[node][split_id] = set(branch_index)
    branch_labels: Dict[str, Dict[str, Set[int]]] = {
        nid: defaultdict(set) for nid in g.nodes
    }
    q: deque[str] = deque()

    # seed: langsung dari XOR-split ke anak-anak
    for s in xor_splits:
        outs = [e for e in g.outgoing.get(s, []) if e.type == "sequenceFlow"]
        for idx, e in enumerate(outs):
            tgt = e.target
            branch_labels[tgt][s].add(idx)
            q.append(tgt)

    # 3. BFS propagate label cabang
    while q:
        v = q.popleft()
        for w in succ.get(v, []):
            changed = False
            for split_id, idx_set in branch_labels[v].items():
                if not idx_set:
                    continue
                old = set(branch_labels[w].get(split_id, set()))
                new = old | idx_set
                if new != old:
                    branch_labels[w][split_id] = new
                    changed = True
            if changed:
                q.append(w)

    # 4. cari AND-join yang punya incoming dari cabang XOR yang beda
    for j, node in g.nodes.items():
        if node.type != "parallelGateway":
            continue
        inc = [s for s in pred.get(j, [])]
        if len(inc) <= 1:
            continue

        is_source_deadlock = False

        for split_id in xor_splits:
            # kumpulkan set index cabang per predecessor via split ini
            sets: List[Set[int]] = []
            for p in inc:
                sset = branch_labels[p].get(split_id, set())
                if sset:
                    sets.append(sset)

            if len(sets) < 2:
                continue

            # cek apakah ada dua set dengan index berbeda
            conflict = False
            base = sets[0]
            for other in sets[1:]:
                if base.isdisjoint(other):
                    # ada index yang berbeda
                    conflict = True
                    break
            if conflict:
                is_source_deadlock = True
                break

        if is_source_deadlock:
            # label di TS: state id-nya sama dengan node id
            if j not in ts.labeling:
                ts.labeling[j] = set()
            ts.labeling[j].add("deadlock")
            ts.labeling[j].add("source_deadlock")

def mark_source_deadlocks(g: BPMNGraph, ts: TransitionSystem):
    """
    Label join (gateway XOR/AND atau activity) yang indegree>1
    tetapi incoming-nya TIDAK semua dikendalikan oleh satu split yang sepadan.
    """
    # 1. Identify candidate join nodes: gateway XOR/AND + activity with indegree >1
    join_candidates = []
    for nid, node in g.nodes.items():
        inc = [e for e in g.incoming.get(nid, []) if e.type == "sequenceFlow"]
        if len(inc) <= 1:
            continue
        if node.type in ("exclusiveGateway", "parallelGateway", "task", "userTask", "serviceTask", "scriptTask", "manualTask", "businessRuleTask"):
            join_candidates.append(nid)

    # 2. For each predecessor, find "controlling split" upstream (last gateway with out>1)
    from collections import deque

    def find_last_split(start: str) -> Optional[str]:
        visited = set()
        q = deque([start])
        last_split = None
        while q:
            v = q.popleft()
            if v in visited:
                continue
            visited.add(v)
            node = g.nodes[v]
            outs = [e for e in g.outgoing.get(v, []) if e.type == "sequenceFlow"]
            ins = [e for e in g.incoming.get(v, []) if e.type == "sequenceFlow"]
            # gateway split jika outgoing >1
            if node.type in ("exclusiveGateway", "parallelGateway") and len(outs) > 1:
                last_split = v
            # terus mundur ke predecessor
            for e in g.incoming.get(v, []):
                if e.type == "sequenceFlow":
                    q.append(e.source)
        return last_split

    for j in join_candidates:
        inc = [e for e in g.incoming.get(j, []) if e.type == "sequenceFlow"]
        splits = []
        for e in inc:
            s = find_last_split(e.source)
            splits.append(s)

        # source-deadlock jika:
        # - ada incoming tanpa split (None), ATAU
        # - ada dua incoming dengan split berbeda
        if any(s is None for s in splits) or len({s for s in splits if s is not None}) > 1:
            ts.labeling.setdefault(j, set()).add("source_deadlock")
            ts.labeling[j].add("deadlock")


def export_graph_to_cypher(g: BPMNGraph, path: str):
    """
    Export node dan edge BPMN ke file Cypher (.cql)
    untuk dibuatkan graph di Neo4j.
    """
    lines: List[str] = []

    # Nodes
    for node in g.nodes.values():
        node_id = json.dumps(node.id)
        node_name = json.dumps(node.name or node.id)
        node_type = json.dumps(node.type)
        process_id = json.dumps(node.process_id) if node.process_id else "null"
        pool = json.dumps(node.pool) if node.pool else "null"

        cyph = (
            f"MERGE (n:BPMNNode {{id: {node_id}}})\n"
            f"SET n.name = {node_name},\n"
            f"    n.type = {node_type},\n"
            f"    n.processId = {process_id},\n"
            f"    n.pool = {pool};"
        )
        lines.append(cyph)

    # Relationships
    for e in g.edges:
        rel_type = "SEQUENCE_FLOW" if e.type == "sequenceFlow" else "MESSAGE_FLOW"
        src = json.dumps(e.source)
        tgt = json.dumps(e.target)
        eid = json.dumps(e.id)
        ename = json.dumps(e.attrs.get("name", e.id))

        cyph = (
            f"MATCH (s:BPMNNode {{id: {src}}}), (t:BPMNNode {{id: {tgt}}})\n"
            f"MERGE (s)-[r:{rel_type} {{id: {eid}}}]->(t)\n"
            f"SET r.name = {ename};"
        )
        lines.append(cyph)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines))


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BPMN + LTL checker with graph analysis.")
    parser.add_argument("--bpmn", required=True, help="Path to BPMN 2.0 XML file")
    parser.add_argument("--props", required=True, help="Path to LTL property file (name = formula)")
    parser.add_argument("--model-name", default="BPMN MODEL", help="Model name for report header")
    parser.add_argument("--out-json", default=None, help="Path to save JSON results")
    parser.add_argument("--out-cypher", default=None, help="Path to save Cypher script for Neo4j")
    parser.add_argument("--out-graph-cypher", default=None, help="Path to save Cypher script to create BPMN graph in Neo4j")
    parser.add_argument("--max-depth", type=int, default=50, help="Max path depth for LTL checking")
    args = parser.parse_args()

    # 1. Parse BPMN and build graph
    g = parse_bpmn_xml(args.bpmn)

    livelock_sccs = detect_livelock_structural(g)

    if livelock_sccs:
        print("\n[Extra] Structural livelock detected:")
        for i, comp in enumerate(livelock_sccs, 1):
            names = [g.nodes[nid].name or nid for nid in comp]
            print(f"  - SCC #{i}: " + ", ".join(names))
    else:
        print("\n[Extra] No structural livelock detected.")

    # 1b. Export graph to Cypher if requested
    if args.out_graph_cypher:
        export_graph_to_cypher(g, args.out_graph_cypher)

    # 2. Build transition system (Kripke structure)
    ts = bpmn_to_transition_system(g)

    # 2b. Enrich labels for LTL properties (loop/source/improper/livelock)
    enrich_labels(g, ts)

    mark_conditional_livelock_gateways(g, ts)

    # 2c. Detect improper structuring deadlock (XOR-split -> AND-join)
    # mark_improper_structuring_deadlocks(g, ts)

    # 2d. Detect source deadlock (join tanpa split sepadan)
    # mark_source_deadlocks(g, ts)

    # 3. Check structural/style rules
    structural_violations = check_bpmn_rules(g)

    # 4. Load and parse LTL properties
    raw_props = load_ltl_properties(args.props)
    ltl_results: List[LTLPropertyResult] = []

    for name, fstr in raw_props:
        try:
            f_ast = parse_ltl(fstr)
        except Exception as e:
            ltl_results.append(
                LTLPropertyResult(
                    name=name,
                    formula_str=fstr,
                    formula=Prop(name="__error__"),
                    passed=False,
                    error=str(e),
                )
            )
            continue

        try:
            passed, ce = check_ltl_property(ts, f_ast, max_depth=args.max_depth)
            ltl_results.append(
                LTLPropertyResult(
                    name=name,
                    formula_str=fstr,
                    formula=f_ast,
                    passed=passed,
                    counterexample=ce,
                )
            )
        except Exception as e:
            ltl_results.append(
                LTLPropertyResult(
                    name=name,
                    formula_str=fstr,
                    formula=f_ast,
                    passed=False,
                    error=str(e),
                )
            )

    # 5. Print report
    print_report(args.model_name, ts, structural_violations, ltl_results)

    # 6. Export JSON
    if args.out_json:
        out = []
        for r in ltl_results:
            out.append(
                {
                    "name": r.name,
                    "formula": r.formula_str,
                    "passed": r.passed,
                    "counterexample": r.counterexample,
                    "error": r.error,
                }
            )
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    # 7. Export Cypher
    if args.out_cypher:
        export_violations_to_cypher(structural_violations, ltl_results, args.out_cypher)

if __name__ == "__main__":
    main()
