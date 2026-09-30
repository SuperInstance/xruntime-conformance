// node_runner.mjs — evaluate with quilt-nn's OWN code and print per-cell digests.
//
// NOTE the workaround, and it is a finding rather than an inconvenience:
// `evaluateForward` throws on a FRESH env for any graph containing `grad` cells,
// because it breaks on them and then runs a finiteness check over `undefined`.
// Their own tests pass (12/12) because the training loop warms the env first, so the
// bug is latent -- but it means "forward pass" has an undocumented precondition, and a
// conformance harness has to work around it to get a cold-start answer at all.
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import * as Q from '/tmp/qnn/src/cellgraph.mjs';

const graph = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const weights = JSON.parse(readFileSync(process.argv[3], 'utf8'));
const order = Q.topoOrder(graph, 'insertion');
const net = Q.loadNet(graph);
const env = { ...weights };
for (const c of graph.cells) if (c.kind === "grad" || c.kind === "tick") env[c.id] = 0;

const samples = [[0,0,0],[0,1,1],[1,0,1],[1,1,0]];
const perCell = {};
let total = 0;
for (const [a,b,y] of samples) {
  Q.evaluateForward(net, env, { x: [a,b], y }, order);
  for (const c of graph.cells) {
    if (["input","weight","grad","tick"].includes(c.kind)) continue;
    if (perCell[c.id] === undefined)
      perCell[c.id] = createHash('sha256').update(`f64|8|${Q.f64hex(env[c.id])}`).digest('hex');
  }
  const lc = graph.cells.find(c => c.kind === 'loss');
  total += env[lc.id];
}
const meanLoss = total / samples.length;
console.log(JSON.stringify({
  runtime: 'node',
  mean_loss_f64hex: Q.f64hex(meanLoss),
  loss_sha: Q.lossShaOf(meanLoss),
  per_cell: perCell,
}, null, 1));
