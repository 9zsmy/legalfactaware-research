# Private GitHub upload

The repository is prepared locally on branch `main`. Do not make it public.

## GitHub CLI route

After GitHub connectivity and authentication are available:

```bash
gh auth login --hostname github.com --git-protocol https --web --scopes repo,workflow
gh repo create legalfactaware-research \
  --private \
  --source=. \
  --remote=origin \
  --push \
  --description "Private LegalFactAware research code and experiment artifacts"
```

Verify the outcome:

```bash
gh repo view --json nameWithOwner,visibility,url,defaultBranchRef
git remote -v
git status --short
```

The expected visibility is `PRIVATE` and the expected default branch is `main`.

## Existing empty repository route

If the private repository is created in the GitHub interface first:

```bash
git remote add origin https://github.com/OWNER/legalfactaware-research.git
git push -u origin main
```

Do not initialize the remote with a README, license, or `.gitignore`; the local repository already contains them.
