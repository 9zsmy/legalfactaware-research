# GitHub publication record

The repository is published from local branch `main` at:

<https://github.com/9zsmy/legalfactaware-research>

Its intended visibility is `PUBLIC`. Raw datasets, model weights, credentials,
and raw legal-document predictions remain excluded from version control.

## GitHub CLI route

The repository was created privately, audited, and then explicitly made public.
For a fresh reconstruction, authenticate and create the repository as follows:

```bash
gh auth login --hostname github.com --git-protocol https --web --scopes repo,workflow
gh repo create legalfactaware-research \
  --public \
  --source=. \
  --remote=origin \
  --push \
  --description "LegalFactAware research code and experiment artifacts"
```

Verify the outcome:

```bash
gh repo view --json nameWithOwner,visibility,url,defaultBranchRef
git remote -v
git status --short
```

The expected visibility is `PUBLIC` and the expected default branch is `main`.

## Existing empty repository route

If an empty repository is created in the GitHub interface first:

```bash
git remote add origin https://github.com/OWNER/legalfactaware-research.git
git push -u origin main
```

Do not initialize the remote with a README, license, or `.gitignore`; the local repository already contains them.
