// TuGraph Browser / lgraph_cli sanity checks
MATCH (n:Endpoint) RETURN count(n) AS endpoints;
MATCH ()-[e:COMMUNICATES]->() RETURN count(e) AS hcg_edges;
MATCH (f:Flow) RETURN count(f) AS flows;

// TCG edge counts by type
MATCH ()-[e:CAUSAL_CR]->() RETURN 'CR' AS type, count(e) AS count
UNION ALL
MATCH ()-[e:CAUSAL_PR]->() RETURN 'PR' AS type, count(e) AS count
UNION ALL
MATCH ()-[e:CAUSAL_DHR]->() RETURN 'DHR' AS type, count(e) AS count
UNION ALL
MATCH ()-[e:CAUSAL_SHR]->() RETURN 'SHR' AS type, count(e) AS count;

// Top application protocols on HCG edges
MATCH ()-[e:COMMUNICATES]->()
RETURN e.protocol_name AS protocol, count(e) AS flows
ORDER BY flows DESC
LIMIT 10;

// High-degree endpoints
MATCH (n:Endpoint)-[e:COMMUNICATES]-()
RETURN n.endpoint_id AS endpoint, count(e) AS degree
ORDER BY degree DESC
LIMIT 20;

// Example CR edges (bidirectional communication)
MATCH (a:Flow)-[e:CAUSAL_CR]->(b:Flow)
RETURN a.flow_id AS src, b.flow_id AS dst, e.src_ip AS src_ip, e.dst_ip AS dst_ip, e.delta_seconds AS delta
LIMIT 10;

// Example PR edges (propagation chain)
MATCH (a:Flow)-[e:CAUSAL_PR]->(b:Flow)
RETURN a.flow_id AS src, b.flow_id AS dst, e.shared_ip AS shared_ip, e.delta_seconds AS delta
LIMIT 10;

// SHR edges (same IP+port, potential port scanning)
MATCH (a:Flow)-[e:CAUSAL_SHR]->(b:Flow)
RETURN a.flow_id AS src, b.flow_id AS dst, e.shared_ip AS ip, e.shared_port AS port, e.delta_seconds AS delta
LIMIT 10;
