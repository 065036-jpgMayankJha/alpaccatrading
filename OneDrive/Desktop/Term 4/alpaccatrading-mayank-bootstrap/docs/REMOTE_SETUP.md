# Push this bootstrap to the GitHub repository

The connected GitHub integration could read `065036-jpgMayankJha/alpaccatrading` but its current installation returned HTTP 403 for repository content writes. This bundle is therefore prepared locally and has **not** been pushed to GitHub yet.

After extracting this bundle into a local clone of `https://github.com/065036-jpgMayankJha/alpaccatrading`, run the following from the repository root:

```bash
git switch -c feature/mayank-agent-strategy
git add .
git commit -m "chore: bootstrap Mayank strategy and risk foundation"
git push -u origin feature/mayank-agent-strategy
```

If your clone has no commits yet, these commands still work after the files have been extracted. If `git switch -c` reports that the branch already exists, switch to it instead.

To permit direct setup through the connected GitHub app in future, grant that integration access to the **personal account** `065036-jpgMayankJha` and the `alpaccatrading` repository. The available installation currently lists the `Setidure-Technologies` organization, which explains why repository reads work but writes are forbidden.
