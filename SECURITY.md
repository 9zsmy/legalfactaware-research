# Security and private-repository policy

## Reporting

Treat this repository as private research material. If a credential, private legal text, restricted dataset row, or model license violation is found, stop sharing the repository and remove the affected object from Git history before further use.

## Before pushing

Run:

```bash
make audit
git status --short
git diff --cached --stat
```

The audit checks common credential patterns, private-key material, model-weight extensions, raw JSONL files, and unusually large files. It is a guardrail, not a substitute for manual review.

## Credentials

- Never commit passwords, access tokens, SSH private keys, `.env` files, cloud credentials, or authenticated URLs.
- Use the operating-system keychain, GitHub credential helper, SSH agent, or GitHub connector.
- Rotate any credential that has appeared in a chat, shell history, log, notebook, or shared archive.

## Private data

Private-repository visibility is not a license to upload datasets or confidential judgments. Dataset terms and personal-data restrictions still apply.

