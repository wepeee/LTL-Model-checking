MERGE (n:BPMNNode {id: "sid-0E555D09-6412-49CD-860E-8C9395048C6C"})
SET n.name = "Close Dispute (No Dispute)",
    n.type = "task",
    n.processId = "sid-34D68456-D661-4F65-BEBB-87561AF2E4CE",
    n.pool = "Marketplace Dispute Management";

MERGE (n:BPMNNode {id: "Activity_0ljw1ck"})
SET n.name = "Close Dispute (Forced Refund)",
    n.type = "task",
    n.processId = "sid-34D68456-D661-4F65-BEBB-87561AF2E4CE",
    n.pool = "Marketplace Dispute Management";

MERGE (n:BPMNNode {id: "sid-55BA467E-3BB5-48A7-A30E-F3116416F5FA"})
SET n.name = "Close Dispute (Dispute Resolved)",
    n.type = "task",
    n.processId = "sid-34D68456-D661-4F65-BEBB-87561AF2E4CE",
    n.pool = "Marketplace Dispute Management";

MERGE (n:BPMNNode {id: "sid-1B321FD9-E6EA-4A6F-9580-C25E9C66A8C7"})
SET n.name = "Process Refund to Customer",
    n.type = "task",
    n.processId = "sid-34D68456-D661-4F65-BEBB-87561AF2E4CE",
    n.pool = "Marketplace Dispute Management";

MERGE (n:BPMNNode {id: "sid-AC9BDC9B-F715-4AE5-B815-8B092D634135"})
SET n.name = "Review Seller's Evidence",
    n.type = "task",
    n.processId = "sid-34D68456-D661-4F65-BEBB-87561AF2E4CE",
    n.pool = "Marketplace Dispute Management";

MERGE (n:BPMNNode {id: "sid-85361256-276A-4D7F-968C-D5CD23E32E05"})
SET n.name = "Force Refund from Seller Payout",
    n.type = "task",
    n.processId = "sid-34D68456-D661-4F65-BEBB-87561AF2E4CE",
    n.pool = "Marketplace Dispute Management";

MERGE (n:BPMNNode {id: "sid-DB65FB9B-5A03-4F6E-99F2-E6BBCBF1A38B"})
SET n.name = "Prepare Refund Request",
    n.type = "task",
    n.processId = "sid-34D68456-D661-4F65-BEBB-87561AF2E4CE",
    n.pool = "Marketplace Dispute Management";

MERGE (n:BPMNNode {id: "sid-45154B7D-E60D-4AC0-8798-5466CBAC6826"})
SET n.name = "Review Order & Dispute Claim",
    n.type = "task",
    n.processId = "sid-34D68456-D661-4F65-BEBB-87561AF2E4CE",
    n.pool = "Marketplace Dispute Management";

MERGE (n:BPMNNode {id: "Activity_0wzxvsr"})
SET n.name = "merge",
    n.type = "task",
    n.processId = "sid-34D68456-D661-4F65-BEBB-87561AF2E4CE",
    n.pool = "Marketplace Dispute Management";

MERGE (n:BPMNNode {id: "Activity_0eld665"})
SET n.name = "merge lagi",
    n.type = "task",
    n.processId = "sid-34D68456-D661-4F65-BEBB-87561AF2E4CE",
    n.pool = "Marketplace Dispute Management";

MERGE (n:BPMNNode {id: "sid-4C197418-25E9-430E-B710-B79A46C0BFCE"})
SET n.name = "Dispute Filed",
    n.type = "startEvent",
    n.processId = "sid-34D68456-D661-4F65-BEBB-87561AF2E4CE",
    n.pool = "Marketplace Dispute Management";

MERGE (n:BPMNNode {id: "sid-79AC0A43-3249-4477-9904-886EF36B3790"})
SET n.name = "end event",
    n.type = "endEvent",
    n.processId = "sid-34D68456-D661-4F65-BEBB-87561AF2E4CE",
    n.pool = "Marketplace Dispute Management";

MERGE (n:BPMNNode {id: "sid-51F0BC71-06FC-4FF6-87FE-3739F69C0D77"})
SET n.name = "Seller Disputes Chargeback",
    n.type = "intermediateCatchEvent",
    n.processId = "sid-34D68456-D661-4F65-BEBB-87561AF2E4CE",
    n.pool = "Marketplace Dispute Management";

MERGE (n:BPMNNode {id: "sid-8CB25F7B-4118-427A-876D-CCD6CA508A12"})
SET n.name = "Seller Response Deadline",
    n.type = "intermediateCatchEvent",
    n.processId = "sid-34D68456-D661-4F65-BEBB-87561AF2E4CE",
    n.pool = "Marketplace Dispute Management";

MERGE (n:BPMNNode {id: "sid-A8177B97-F00F-4215-BE08-5A98DCD6E33F"})
SET n.name = "Seller Accepts Chargeback",
    n.type = "intermediateCatchEvent",
    n.processId = "sid-34D68456-D661-4F65-BEBB-87561AF2E4CE",
    n.pool = "Marketplace Dispute Management";

MERGE (n:BPMNNode {id: "sid-A513CA88-D80D-47DD-AB50-D49731BF642D"})
SET n.name = "Notify Seller of Chargeback",
    n.type = "intermediateThrowEvent",
    n.processId = "sid-34D68456-D661-4F65-BEBB-87561AF2E4CE",
    n.pool = "Marketplace Dispute Management";

MERGE (n:BPMNNode {id: "sid-E0333AC6-E277-475E-BCFF-609F44E18CA9"})
SET n.name = "Seller's Evidence Valid?",
    n.type = "exclusiveGateway",
    n.processId = "sid-34D68456-D661-4F65-BEBB-87561AF2E4CE",
    n.pool = "Marketplace Dispute Management";

MERGE (n:BPMNNode {id: "sid-4149A4F0-8398-4CFD-A2C8-67D4CC2689A1"})
SET n.name = "Seller at Fault?",
    n.type = "exclusiveGateway",
    n.processId = "sid-34D68456-D661-4F65-BEBB-87561AF2E4CE",
    n.pool = "Marketplace Dispute Management";

MERGE (n:BPMNNode {id: "Gateway_1n9rmrt"})
SET n.name = "Gateway_1n9rmrt",
    n.type = "parallelGateway",
    n.processId = "sid-34D68456-D661-4F65-BEBB-87561AF2E4CE",
    n.pool = "Marketplace Dispute Management";

MERGE (n:BPMNNode {id: "Gateway_1jp0xgo"})
SET n.name = "Gateway_1jp0xgo",
    n.type = "parallelGateway",
    n.processId = "sid-34D68456-D661-4F65-BEBB-87561AF2E4CE",
    n.pool = "Marketplace Dispute Management";

MERGE (n:BPMNNode {id: "sid-5191B77D-3008-4E49-85D9-F8297FAFCAD1"})
SET n.name = "Wait for Seller Response",
    n.type = "eventBasedGateway",
    n.processId = "sid-34D68456-D661-4F65-BEBB-87561AF2E4CE",
    n.pool = "Marketplace Dispute Management";

MATCH (s:BPMNNode {id: "Gateway_1jp0xgo"}), (t:BPMNNode {id: "sid-0E555D09-6412-49CD-860E-8C9395048C6C"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "Flow_0tdq54z"}]->(t)
SET r.name = "Flow_0tdq54z";

MATCH (s:BPMNNode {id: "Gateway_1n9rmrt"}), (t:BPMNNode {id: "sid-79AC0A43-3249-4477-9904-886EF36B3790"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "Flow_1nztzhl"}]->(t)
SET r.name = "Flow_1nztzhl";

MATCH (s:BPMNNode {id: "sid-0E555D09-6412-49CD-860E-8C9395048C6C"}), (t:BPMNNode {id: "Activity_0eld665"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "Flow_15qi1dp"}]->(t)
SET r.name = "Flow_15qi1dp";

MATCH (s:BPMNNode {id: "Activity_0ljw1ck"}), (t:BPMNNode {id: "Activity_0eld665"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "Flow_06q4pz3"}]->(t)
SET r.name = "Flow_06q4pz3";

MATCH (s:BPMNNode {id: "sid-55BA467E-3BB5-48A7-A30E-F3116416F5FA"}), (t:BPMNNode {id: "Activity_0eld665"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "sid-5E51064C-CD42-48D4-A97F-90D3F4D87C1F"}]->(t)
SET r.name = "sid-5E51064C-CD42-48D4-A97F-90D3F4D87C1F";

MATCH (s:BPMNNode {id: "sid-1B321FD9-E6EA-4A6F-9580-C25E9C66A8C7"}), (t:BPMNNode {id: "sid-55BA467E-3BB5-48A7-A30E-F3116416F5FA"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "sid-F242D4D1-8026-4870-8F57-96F148E82A58"}]->(t)
SET r.name = "sid-F242D4D1-8026-4870-8F57-96F148E82A58";

MATCH (s:BPMNNode {id: "sid-E0333AC6-E277-475E-BCFF-609F44E18CA9"}), (t:BPMNNode {id: "Activity_0wzxvsr"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "Flow_14awd2i"}]->(t)
SET r.name = "Yes";

MATCH (s:BPMNNode {id: "sid-AC9BDC9B-F715-4AE5-B815-8B092D634135"}), (t:BPMNNode {id: "sid-E0333AC6-E277-475E-BCFF-609F44E18CA9"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "sid-9B09B900-5DF1-4479-AD51-E72B44E8E373"}]->(t)
SET r.name = "sid-9B09B900-5DF1-4479-AD51-E72B44E8E373";

MATCH (s:BPMNNode {id: "sid-51F0BC71-06FC-4FF6-87FE-3739F69C0D77"}), (t:BPMNNode {id: "sid-AC9BDC9B-F715-4AE5-B815-8B092D634135"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "sid-F44C6137-EE20-421D-9AC8-3510B35434A3"}]->(t)
SET r.name = "sid-F44C6137-EE20-421D-9AC8-3510B35434A3";

MATCH (s:BPMNNode {id: "sid-5191B77D-3008-4E49-85D9-F8297FAFCAD1"}), (t:BPMNNode {id: "sid-51F0BC71-06FC-4FF6-87FE-3739F69C0D77"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "sid-6548CEB8-DF10-4A02-A2C7-F4B5CE4C7A29"}]->(t)
SET r.name = "sid-6548CEB8-DF10-4A02-A2C7-F4B5CE4C7A29";

MATCH (s:BPMNNode {id: "sid-85361256-276A-4D7F-968C-D5CD23E32E05"}), (t:BPMNNode {id: "Activity_0ljw1ck"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "Flow_0z9gpyi"}]->(t)
SET r.name = "Flow_0z9gpyi";

MATCH (s:BPMNNode {id: "sid-E0333AC6-E277-475E-BCFF-609F44E18CA9"}), (t:BPMNNode {id: "sid-85361256-276A-4D7F-968C-D5CD23E32E05"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "sid-442A4255-0804-4E3E-A17F-AAB58040BD52"}]->(t)
SET r.name = "No";

MATCH (s:BPMNNode {id: "sid-8CB25F7B-4118-427A-876D-CCD6CA508A12"}), (t:BPMNNode {id: "sid-85361256-276A-4D7F-968C-D5CD23E32E05"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "sid-1E9CD56C-E0A6-4BFD-AAF8-F5EF1DA84198"}]->(t)
SET r.name = "sid-1E9CD56C-E0A6-4BFD-AAF8-F5EF1DA84198";

MATCH (s:BPMNNode {id: "sid-5191B77D-3008-4E49-85D9-F8297FAFCAD1"}), (t:BPMNNode {id: "sid-8CB25F7B-4118-427A-876D-CCD6CA508A12"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "sid-42617F50-E144-4099-8913-93D49F8512F2"}]->(t)
SET r.name = "sid-42617F50-E144-4099-8913-93D49F8512F2";

MATCH (s:BPMNNode {id: "sid-A8177B97-F00F-4215-BE08-5A98DCD6E33F"}), (t:BPMNNode {id: "sid-1B321FD9-E6EA-4A6F-9580-C25E9C66A8C7"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "sid-484E3A80-DACF-46AD-BA41-5AF7190D146B"}]->(t)
SET r.name = "sid-484E3A80-DACF-46AD-BA41-5AF7190D146B";

MATCH (s:BPMNNode {id: "sid-5191B77D-3008-4E49-85D9-F8297FAFCAD1"}), (t:BPMNNode {id: "sid-A8177B97-F00F-4215-BE08-5A98DCD6E33F"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "sid-10B31FA5-661D-475B-AF31-4A72FCB8B57B"}]->(t)
SET r.name = "sid-10B31FA5-661D-475B-AF31-4A72FCB8B57B";

MATCH (s:BPMNNode {id: "sid-A513CA88-D80D-47DD-AB50-D49731BF642D"}), (t:BPMNNode {id: "sid-5191B77D-3008-4E49-85D9-F8297FAFCAD1"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "sid-ECABAB61-45A1-48B1-8046-2A378F2361EF"}]->(t)
SET r.name = "sid-ECABAB61-45A1-48B1-8046-2A378F2361EF";

MATCH (s:BPMNNode {id: "sid-DB65FB9B-5A03-4F6E-99F2-E6BBCBF1A38B"}), (t:BPMNNode {id: "sid-A513CA88-D80D-47DD-AB50-D49731BF642D"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "sid-111F98D4-3B58-4113-91DE-B5D05D0129B4"}]->(t)
SET r.name = "sid-111F98D4-3B58-4113-91DE-B5D05D0129B4";

MATCH (s:BPMNNode {id: "sid-4149A4F0-8398-4CFD-A2C8-67D4CC2689A1"}), (t:BPMNNode {id: "Activity_0wzxvsr"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "sid-3F912482-AE97-467E-826C-978E068C3D76"}]->(t)
SET r.name = "No";

MATCH (s:BPMNNode {id: "sid-4149A4F0-8398-4CFD-A2C8-67D4CC2689A1"}), (t:BPMNNode {id: "sid-DB65FB9B-5A03-4F6E-99F2-E6BBCBF1A38B"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "sid-EADBB819-B706-4DC4-BE23-14076516395F"}]->(t)
SET r.name = "Yes";

MATCH (s:BPMNNode {id: "sid-45154B7D-E60D-4AC0-8798-5466CBAC6826"}), (t:BPMNNode {id: "sid-4149A4F0-8398-4CFD-A2C8-67D4CC2689A1"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "sid-24193134-FAFD-45C9-9587-DE4F921198B7"}]->(t)
SET r.name = "sid-24193134-FAFD-45C9-9587-DE4F921198B7";

MATCH (s:BPMNNode {id: "sid-4C197418-25E9-430E-B710-B79A46C0BFCE"}), (t:BPMNNode {id: "sid-45154B7D-E60D-4AC0-8798-5466CBAC6826"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "sid-997D803C-519F-419C-9C86-BD6EFB84806F"}]->(t)
SET r.name = "sid-997D803C-519F-419C-9C86-BD6EFB84806F";

MATCH (s:BPMNNode {id: "Activity_0wzxvsr"}), (t:BPMNNode {id: "Gateway_1jp0xgo"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "Flow_14f3jvk"}]->(t)
SET r.name = "Flow_14f3jvk";

MATCH (s:BPMNNode {id: "Activity_0eld665"}), (t:BPMNNode {id: "Gateway_1n9rmrt"})
MERGE (s)-[r:SEQUENCE_FLOW {id: "Flow_0bnvw5t"}]->(t)
SET r.name = "Flow_0bnvw5t";