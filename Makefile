.PHONY: syntax test-recent audit test

PYTHON ?= python3

syntax:
	PYTHONPYCACHEPREFIX=.cache/pycache $(PYTHON) -m compileall -q src evaluation baselines experiments tools

test-recent:
	$(PYTHON) -m unittest discover -s experiments/recent_baselines/tests -v

audit:
	$(PYTHON) tools/audit_repository.py .

test: syntax test-recent audit

