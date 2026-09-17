"""v2 follow-up: make the donor key unconditionally cohort_uid on BOTH
__getitem__ paths in src/data/dataset.py (the opt-in _physical_pids branch
never runs by default, so the live path was hashing patient_uid).

    python scripts/v2_patch_pid.py
"""
import io, shutil, sys
p = "src/data/dataset.py"
t = io.open(p, encoding="utf-8").read()

def rep(old, new, label):
    global t
    if new in t and old not in t:
        print(f"  = already applied: {label}"); return
    if t.count(old) != 1:
        i = t.find(old[:50])
        print(f"*** {label}: pattern {'missing' if t.count(old)==0 else 'ambiguous'}\n--- context ---\n"
              f"{t[max(0,i-200):i+500] if i>=0 else '(not found)'}"); sys.exit(1)
    t = t.replace(old, new, 1); print(f"  + {label}")

# live path (lines ~307-313)
rep('''            if self._physical_pids is not None:
                out["pid"] = torch.tensor(_stable_hash(str(r["cohort_uid"])), dtype=torch.long)  # v2: COHORT key
                out["uid"] = str(r.get("sample_uid", r.get("filepath", i)))
            else:
                _pu = str(r.get("patient_uid", r.get("patient_id", i)))
                out["pid"] = torch.tensor(zlib.crc32(_pu.encode()) & 0x7FFFFFFF,
                                          dtype=torch.long)
            return out''',
'''            # v2: donor key is ALWAYS the physical-patient (cohort) key
            out["pid"] = torch.tensor(_stable_hash(str(r["cohort_uid"])), dtype=torch.long)
            out["uid"] = str(r.get("sample_uid", r.get("filepath", i)))
            return out''', "live path: unconditional cohort pid")

# earlier physics block (lines ~235-238) — keep consistent if ever reached
rep('''                _pu = str(r.get('patient_uid', r.get('patient_id', i)))
                out['pid'] = torch.tensor(
                    zlib.crc32(_pu.encode()) & 0x7FFFFFFF, dtype=torch.long)
                return out''',
'''                out['pid'] = torch.tensor(_stable_hash(str(r["cohort_uid"])), dtype=torch.long)  # v2
                out['uid'] = str(r.get('sample_uid', r.get('filepath', i)))
                return out''', "physics block: cohort pid")

shutil.copy(p, p + ".v2bak2")
io.open(p, "w", encoding="utf-8").write(t)
print("done — rerun: python scripts/v2_gates.py")