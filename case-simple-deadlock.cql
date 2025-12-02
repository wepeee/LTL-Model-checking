MATCH (n:BPMNNode {id: "sid-4C197418-25E9-430E-B710-B79A46C0BFCE"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_source_deadlock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_source_deadlock: G ( ((exXORsplit) && ! F (exXORjoin)) -> F (deadlock) )";

MATCH (n:BPMNNode {id: "sid-45154B7D-E60D-4AC0-8798-5466CBAC6826"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_source_deadlock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_source_deadlock: G ( ((exXORsplit) && ! F (exXORjoin)) -> F (deadlock) )";

MATCH (n:BPMNNode {id: "sid-4149A4F0-8398-4CFD-A2C8-67D4CC2689A1"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_source_deadlock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_source_deadlock: G ( ((exXORsplit) && ! F (exXORjoin)) -> F (deadlock) )";

MATCH (n:BPMNNode {id: "sid-DB65FB9B-5A03-4F6E-99F2-E6BBCBF1A38B"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_source_deadlock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_source_deadlock: G ( ((exXORsplit) && ! F (exXORjoin)) -> F (deadlock) )";

MATCH (n:BPMNNode {id: "sid-A513CA88-D80D-47DD-AB50-D49731BF642D"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_source_deadlock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_source_deadlock: G ( ((exXORsplit) && ! F (exXORjoin)) -> F (deadlock) )";

MATCH (n:BPMNNode {id: "sid-5191B77D-3008-4E49-85D9-F8297FAFCAD1"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_source_deadlock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_source_deadlock: G ( ((exXORsplit) && ! F (exXORjoin)) -> F (deadlock) )";

MATCH (n:BPMNNode {id: "sid-8CB25F7B-4118-427A-876D-CCD6CA508A12"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_source_deadlock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_source_deadlock: G ( ((exXORsplit) && ! F (exXORjoin)) -> F (deadlock) )";

MATCH (n:BPMNNode {id: "sid-85361256-276A-4D7F-968C-D5CD23E32E05"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_source_deadlock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_source_deadlock: G ( ((exXORsplit) && ! F (exXORjoin)) -> F (deadlock) )";

MATCH (n:BPMNNode {id: "Activity_0ljw1ck"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_source_deadlock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_source_deadlock: G ( ((exXORsplit) && ! F (exXORjoin)) -> F (deadlock) )";

MATCH (n:BPMNNode {id: "Gateway_1n9rmrt"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_source_deadlock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_source_deadlock: G ( ((exXORsplit) && ! F (exXORjoin)) -> F (deadlock) )";

MATCH (n:BPMNNode {id: "sid-79AC0A43-3249-4477-9904-886EF36B3790"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_source_deadlock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_source_deadlock: G ( ((exXORsplit) && ! F (exXORjoin)) -> F (deadlock) )";

MATCH (n:BPMNNode {id: "sid-4C197418-25E9-430E-B710-B79A46C0BFCE"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_improper_structuring_deadlock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_improper_structuring_deadlock: G ( (((exXORsplit && path1) && !path2) && F (ANDjoin_waiting)) -> F (deadlock) )";

MATCH (n:BPMNNode {id: "sid-45154B7D-E60D-4AC0-8798-5466CBAC6826"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_improper_structuring_deadlock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_improper_structuring_deadlock: G ( (((exXORsplit && path1) && !path2) && F (ANDjoin_waiting)) -> F (deadlock) )";

MATCH (n:BPMNNode {id: "sid-4149A4F0-8398-4CFD-A2C8-67D4CC2689A1"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_improper_structuring_deadlock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_improper_structuring_deadlock: G ( (((exXORsplit && path1) && !path2) && F (ANDjoin_waiting)) -> F (deadlock) )";

MATCH (n:BPMNNode {id: "sid-DB65FB9B-5A03-4F6E-99F2-E6BBCBF1A38B"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_improper_structuring_deadlock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_improper_structuring_deadlock: G ( (((exXORsplit && path1) && !path2) && F (ANDjoin_waiting)) -> F (deadlock) )";

MATCH (n:BPMNNode {id: "sid-A513CA88-D80D-47DD-AB50-D49731BF642D"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_improper_structuring_deadlock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_improper_structuring_deadlock: G ( (((exXORsplit && path1) && !path2) && F (ANDjoin_waiting)) -> F (deadlock) )";

MATCH (n:BPMNNode {id: "sid-5191B77D-3008-4E49-85D9-F8297FAFCAD1"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_improper_structuring_deadlock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_improper_structuring_deadlock: G ( (((exXORsplit && path1) && !path2) && F (ANDjoin_waiting)) -> F (deadlock) )";

MATCH (n:BPMNNode {id: "sid-8CB25F7B-4118-427A-876D-CCD6CA508A12"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_improper_structuring_deadlock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_improper_structuring_deadlock: G ( (((exXORsplit && path1) && !path2) && F (ANDjoin_waiting)) -> F (deadlock) )";

MATCH (n:BPMNNode {id: "sid-85361256-276A-4D7F-968C-D5CD23E32E05"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_improper_structuring_deadlock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_improper_structuring_deadlock: G ( (((exXORsplit && path1) && !path2) && F (ANDjoin_waiting)) -> F (deadlock) )";

MATCH (n:BPMNNode {id: "Activity_0ljw1ck"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_improper_structuring_deadlock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_improper_structuring_deadlock: G ( (((exXORsplit && path1) && !path2) && F (ANDjoin_waiting)) -> F (deadlock) )";

MATCH (n:BPMNNode {id: "Gateway_1n9rmrt"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_improper_structuring_deadlock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_improper_structuring_deadlock: G ( (((exXORsplit && path1) && !path2) && F (ANDjoin_waiting)) -> F (deadlock) )";

MATCH (n:BPMNNode {id: "sid-79AC0A43-3249-4477-9904-886EF36B3790"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_improper_structuring_deadlock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_improper_structuring_deadlock: G ( (((exXORsplit && path1) && !path2) && F (ANDjoin_waiting)) -> F (deadlock) )";

MATCH (n:BPMNNode {id: "sid-4C197418-25E9-430E-B710-B79A46C0BFCE"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_conditional_livelock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_conditional_livelock: G ( (exGateway1 && exGateway2) -> F ( ! (F Final) ) )";

MATCH (n:BPMNNode {id: "sid-45154B7D-E60D-4AC0-8798-5466CBAC6826"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_conditional_livelock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_conditional_livelock: G ( (exGateway1 && exGateway2) -> F ( ! (F Final) ) )";

MATCH (n:BPMNNode {id: "sid-4149A4F0-8398-4CFD-A2C8-67D4CC2689A1"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_conditional_livelock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_conditional_livelock: G ( (exGateway1 && exGateway2) -> F ( ! (F Final) ) )";

MATCH (n:BPMNNode {id: "sid-DB65FB9B-5A03-4F6E-99F2-E6BBCBF1A38B"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_conditional_livelock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_conditional_livelock: G ( (exGateway1 && exGateway2) -> F ( ! (F Final) ) )";

MATCH (n:BPMNNode {id: "sid-A513CA88-D80D-47DD-AB50-D49731BF642D"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_conditional_livelock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_conditional_livelock: G ( (exGateway1 && exGateway2) -> F ( ! (F Final) ) )";

MATCH (n:BPMNNode {id: "sid-5191B77D-3008-4E49-85D9-F8297FAFCAD1"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_conditional_livelock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_conditional_livelock: G ( (exGateway1 && exGateway2) -> F ( ! (F Final) ) )";

MATCH (n:BPMNNode {id: "sid-8CB25F7B-4118-427A-876D-CCD6CA508A12"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_conditional_livelock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_conditional_livelock: G ( (exGateway1 && exGateway2) -> F ( ! (F Final) ) )";

MATCH (n:BPMNNode {id: "sid-85361256-276A-4D7F-968C-D5CD23E32E05"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_conditional_livelock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_conditional_livelock: G ( (exGateway1 && exGateway2) -> F ( ! (F Final) ) )";

MATCH (n:BPMNNode {id: "Activity_0ljw1ck"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_conditional_livelock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_conditional_livelock: G ( (exGateway1 && exGateway2) -> F ( ! (F Final) ) )";

MATCH (n:BPMNNode {id: "Gateway_1n9rmrt"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_conditional_livelock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_conditional_livelock: G ( (exGateway1 && exGateway2) -> F ( ! (F Final) ) )";

MATCH (n:BPMNNode {id: "sid-79AC0A43-3249-4477-9904-886EF36B3790"})
SET n.violated = true,
    n.violated_rules = coalesce(n.violated_rules, []) + "LTL_conditional_livelock",
    n.violation_reasons = coalesce(n.violation_reasons, []) + "LTL_conditional_livelock: G ( (exGateway1 && exGateway2) -> F ( ! (F Final) ) )";