# Contributing

First off, thanks for taking the time to contribute! 🎉

This project is in its early stages, so all types of contributions—including documentation, bug reports, and code suggestions—are highly valued and appreciated.

## How to Contribute

### 1. Reporting Bugs
Before submitting a bug report, please check if the issue has already been reported in the [Issues tab](https://github.com/kmangal/sherlock/issues).

If you find a new bug, please include:
*   A clear, descriptive title.
*   Steps to reproduce the bug.
*   Expected behavior vs. actual behavior.
*   Your environment (OS, language version, etc.).

### 2. Suggesting Enhancements
We welcome ideas for new features or improvements! Please open an issue and tag it as an "enhancement" to start a discussion.

### 3. Pull Requests (Code & Documentation)
1.  **Fork** the repository and create your branch: `git checkout -b your-name/amazing-feature`.
2.  **Make your changes**, following the existing code style.
3.  **Ensure tests pass** (if applicable).
4.  **Submit a pull request** with a clear description of what you changed.

*Note: For early-stage projects, it is often best to discuss large changes in an issue first before writing code.*

### 4. Code of Conduct
By participating in this project, you agree to maintain a respectful and welcoming environment for everyone.

## Setting Up the Development Environment

Make sure that you have both `uv` and `yarn` installed on your machine.

1.  Clone the repo: `git clone https://github.com/kmangal/sherlock.git`
2.  Install `uv` on your machine:

    Linux / MacOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
    Windows Powershell: `irm https://astral.sh/uv/install.ps1 | iex`

3.  Install `pre-commit` hooks: `uv tool install pre-commit && pre-commit install`
4.  Install the backend dependencies: `(cd backend && uv sync --all-groups)`
5.  Build the backend: XXXXX
6.  Install the frontend dependencies: `(cd frontend && yarn install)`
7.  Run using XXXX

## Sytle Guide

We use Conventional Commits.

The format should be:

```
<type>: <short summary>

<optional body>
```

Types to use in this repo:
* feat – new functionality
* fix – bug fix
* docs – documentation only
* chore – tooling, CI, maintenance
* refactor – code change without behavior change
* test – adding or updating tests
* build – build system or dependencies
* ci – CI configuration


## Need Help?
Feel free to contact us by opening an issue or reaching out via email at kmangal@alumni.harvard.edu.

Thank you for helping us grow!