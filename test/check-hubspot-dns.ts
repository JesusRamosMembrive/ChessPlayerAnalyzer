#!/usr/bin/env ts-node

import { promises as dns } from "dns";

type Result = { check: string; ok: boolean; details?: string };

async function checkCname(host: string): Promise<Result> {
  try {
    const res = await dns.resolveCname(host);
    return { check: `CNAME ${host}`, ok: res.length > 0, details: res.join(", ") };
  } catch (e: any) {
    return { check: `CNAME ${host}`, ok: false, details: e.message };
  }
}

async function checkTxt(
  host: string,
  predicate?: (txt: string) => boolean,
  label?: string,
): Promise<Result> {
  try {
    const res = await dns.resolveTxt(host);
    const flat = res.map((chunks) => chunks.join("")).join(" | ");
    const passes = predicate ? predicate(flat) : res.length > 0;
    return { check: `${label ?? "TXT"} ${host}`, ok: passes, details: flat };
  } catch (e: any) {
    return { check: `${label ?? "TXT"} ${host}`, ok: false, details: e.message };
  }
}

async function main() {
  const domain = process.argv[2];
  const selectors = (process.argv[3] ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  if (!domain || selectors.length === 0) {
    console.error("Uso: ts-node check-hubspot-dns.ts <dominio> <selector1,selector2>");
    process.exit(1);
  }

  const checks: Result[] = [];
  for (const s of selectors) checks.push(await checkCname(`${s}._domainkey.${domain}`));

  // SPF: un único TXT que contenga include:*.hubspotemail.net y termine en -all
  checks.push(
    await checkTxt(
      domain,
      (t) => /^v=spf1\s.+-all$/i.test(t) && /hubspotemail\.net/.test(t),
      "SPF",
    ),
  );

  // DMARC: debe empezar por v=DMARC1;
  checks.push(
    await checkTxt(`_dmarc.${domain}`, (t) => /^v=DMARC1;/i.test(t), "DMARC"),
  );

  const ok = checks.every((c) => c.ok);
  for (const c of checks) {
    console.log(`${c.ok ? "✔" : "✖"} ${c.check}${c.details ? ` → ${c.details}` : ""}`);
  }
  process.exit(ok ? 0 : 2);
}

main().catch((e) => {
  console.error(e);
  process.exit(2);
});
