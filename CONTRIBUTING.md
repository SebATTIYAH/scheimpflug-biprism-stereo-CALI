# Contributing

Propose changes with an explanation of the physical assumptions, units,
coordinate frame, and evidence affected.

Run the core and affine test suites plus the relevant synthetic experiment. For
changes to `extensions/brown_v4/`, run its two CLI smoke checks from that
directory in an isolated environment. The core and Brown packages share the
name `bbs_calib` and must not be overlaid.
Keep held-out frames outside camera fitting and model selection. Never substitute
fitted target poses for independent measured geometric truth. Do not remove
invalid points or relax ray/domain checks merely to reduce an RMS value.

New model terms require a new version or explicit compatibility rule. Do not
interchange physical, affine-candidate, or later joint-bundle calibration files.
Record reproducible seeds and commands for numerical results. Keep raw private
data, credentials, machine paths, and generated caches out of source commits.
Run `python tools/audit_git_tree.py` after staging files for a commit.
