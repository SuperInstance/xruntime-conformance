// node_attn_runner.mjs — evaluate with quilt-attention's OWN code, unmodified, and print
// per-cell array digests for the Python side to be diffed against.
import { readFileSync } from 'node:fs';
import * as Q from '/tmp/qatt/src/attncells.mjs';

const graph = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const weights = JSON.parse(readFileSync(process.argv[3], 'utf8'));
const net = Q.loadNet(graph);
const env = {};
for (const [k, v] of Object.entries(weights)) env[k] = Float64Array.from(v);
const samples = JSON.parse(readFileSync(process.argv[4], 'utf8'));

const perCell = {};
for (const s of samples) {
  const w = Q.evaluateForward(net, env, {
    x: Float64Array.from(s.x), y: Float64Array.from(s.y),
  }, net.order, true);
  for (const rec of w) {
    if (perCell[rec.cell] === undefined) perCell[rec.cell] = rec.digest;
  }
}
const lossId = net.lossId;
const ce = env[lossId][0];
console.log(JSON.stringify({
  runtime: 'node',
  loss_f64hex: Q.f64hex(ce),
  scalar_sha_portable: Q.sha256hex(Buffer.from(Q.f64hex(ce), 'hex').toString('latin1')),
  per_cell: perCell,
}, null, 1));
