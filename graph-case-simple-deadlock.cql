MERGE (n:BPMNNode {id: "Activity_1f2xj6c"})
SET n.name = "Terima Pengajuan",
    n.type = "task",
    n.processId = "Process_002rvb4",
    n.pool = "Pengajuan Kredit";

MERGE (n:BPMNNode {id: "Activity_1v0n87a"})
SET n.name = "task a",
    n.type = "task",
    n.processId = "Process_002rvb4",
    n.pool = "Pengajuan Kredit";

MERGE (n:BPMNNode {id: "Activity_1ekq4l7"})
SET n.name = "task b",
    n.type = "task",
    n.processId = "Process_002rvb4",
    n.pool = "Pengajuan Kredit";

MERGE (n:BPMNNode {id: "Event_0qdp6v0"})
SET n.name = "start",
    n.type = "startEvent",
    n.processId = "Process_002rvb4",
    n.pool = "Pengajuan Kredit";

MERGE (n:BPMNNode {id: "Event_048m79m"})
SET n.name = "end",
    n.type = "endEvent",
    n.processId = "Process_002rvb4",
    n.pool = "Pengajuan Kredit";

MERGE (n:BPMNNode {id: "Gateway_13btx37"})
SET n.name = "Gateway_13btx37",
    n.type = "exclusiveGateway",
    n.processId = "Process_002rvb4",
    n.pool = "Pengajuan Kredit";

MERGE (n:BPMNNode {id: "Gateway_0shoetf"})
SET n.name = "Gateway_0shoetf",
    n.type = "exclusiveGateway",
    n.processId = "Process_002rvb4",
    n.pool = "Pengajuan Kredit";

MATCH (s:BPMNNode {id: "Event_0qdp6v0"}), (t:BPMNNode {id: "Activity_1f2xj6c"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "Flow_0gw52ic"}]->(t)
SET r.name = "Flow_0gw52ic";

MATCH (s:BPMNNode {id: "Activity_1f2xj6c"}), (t:BPMNNode {id: "Gateway_13btx37"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "Flow_0qb8kt8"}]->(t)
SET r.name = "Flow_0qb8kt8";

MATCH (s:BPMNNode {id: "Gateway_13btx37"}), (t:BPMNNode {id: "Gateway_0shoetf"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "Flow_1f8v7q1"}]->(t)
SET r.name = "Flow_1f8v7q1";

MATCH (s:BPMNNode {id: "Gateway_0shoetf"}), (t:BPMNNode {id: "Activity_1f2xj6c"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "Flow_1inkpy6"}]->(t)
SET r.name = "Flow_1inkpy6";

MATCH (s:BPMNNode {id: "Gateway_0shoetf"}), (t:BPMNNode {id: "Activity_1v0n87a"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "Flow_1esyjs4"}]->(t)
SET r.name = "Flow_1esyjs4";

MATCH (s:BPMNNode {id: "Activity_1v0n87a"}), (t:BPMNNode {id: "Event_048m79m"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "Flow_1pt5doy"}]->(t)
SET r.name = "Flow_1pt5doy";

MATCH (s:BPMNNode {id: "Gateway_13btx37"}), (t:BPMNNode {id: "Activity_1ekq4l7"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "Flow_00sjy7t"}]->(t)
SET r.name = "Flow_00sjy7t";

MATCH (s:BPMNNode {id: "Activity_1ekq4l7"}), (t:BPMNNode {id: "Event_048m79m"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "Flow_15449q9"}]->(t)
SET r.name = "Flow_15449q9";