"""Action-first Python entry point for Bash, PowerShell and other terminals."""
import os
import sys

# Configure numeric runtime before importing numpy/scipy through the CLI.
from bbs_calib.runtime import configure
configure()
if hasattr(sys.stdout,'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

if __name__=='__main__':
    from bbs_calib.__main__ import main
    sys.exit(main())
