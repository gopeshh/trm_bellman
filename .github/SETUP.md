# Repository Setup Guide

This document provides instructions for setting up the repository with proper labels, branch protection, and workflows.

## Issue Labels Setup

Create the following labels in your GitHub repository settings:

### How to Add Labels
1. Go to repository **Settings** → **Issues** → **Labels**
2. Click **New label** for each label below
3. Add the name, description, and color

### Label Definitions

#### Area Labels (Color: #0075ca - Blue)
- `area:rl` - Reinforcement learning related work
- `area:training` - Training infrastructure and optimization
- `area:models` - Model architecture changes
- `area:dataset` - Dataset generation or processing

#### Task Labels (Color: #7057ff - Purple)
- `task:sudoku` - Sudoku puzzle solving
- `task:maze` - Maze navigation
- `task:arc` - ARC (Abstraction and Reasoning Corpus) challenges

#### Priority Labels
- `priority:p0` - Critical, blocking issues (Color: #d73a4a - Red)
- `priority:p1` - High priority (Color: #ff9800 - Orange)
- `priority:p2` - Medium priority (Color: #fbca04 - Yellow)
- `priority:p3` - Low priority, nice to have (Color: #c5def5 - Light Blue)

#### Type Labels (Color: #008672 - Green)
- `type:bug` - Bug reports
- `type:feature` - Feature requests
- `type:docs` - Documentation improvements
- `type:question` - Questions or discussions

### Quick Setup Script

You can also use the GitHub CLI to create labels:

```bash
# Area labels
gh label create "area:rl" --description "Reinforcement learning related work" --color 0075ca
gh label create "area:training" --description "Training infrastructure and optimization" --color 0075ca
gh label create "area:models" --description "Model architecture changes" --color 0075ca
gh label create "area:dataset" --description "Dataset generation or processing" --color 0075ca

# Task labels
gh label create "task:sudoku" --description "Sudoku puzzle solving" --color 7057ff
gh label create "task:maze" --description "Maze navigation" --color 7057ff
gh label create "task:arc" --description "ARC challenges" --color 7057ff

# Priority labels
gh label create "priority:p0" --description "Critical, blocking issues" --color d73a4a
gh label create "priority:p1" --description "High priority" --color ff9800
gh label create "priority:p2" --description "Medium priority" --color fbca04
gh label create "priority:p3" --description "Low priority, nice to have" --color c5def5

# Type labels
gh label create "type:bug" --description "Bug reports" --color 008672
gh label create "type:feature" --description "Feature requests" --color 008672
gh label create "type:docs" --description "Documentation improvements" --color 008672
gh label create "type:question" --description "Questions or discussions" --color 008672
```

## Branch Protection Setup

### Configure Branch Protection for `main`

1. Go to repository **Settings** → **Branches**
2. Click **Add branch protection rule**
3. Enter branch name pattern: `main`
4. Enable the following settings:

#### Required Settings
- ✅ **Require a pull request before merging**
  - Require approvals: 1 (or more if desired)
  - Dismiss stale pull request approvals when new commits are pushed

- ✅ **Require status checks to pass before merging**
  - Require branches to be up to date before merging
  - Add status checks: `Unit Tests`, `Code Quality` (these come from the CI workflow)

- ✅ **Do not allow bypassing the above settings**

#### Optional but Recommended
- ✅ Require conversation resolution before merging
- ✅ Require linear history
- ✅ Include administrators (applies rules to admins too)

### GitHub Actions Workflow

The CI workflow (`.github/workflows/ci.yml`) is already configured and will:
- Run automatically on pull requests to `main`
- Execute unit tests in the `tests/` directory
- Check code quality with linters
- Provide status checks for branch protection

**Note**: If you don't have a `tests/` directory yet, the workflow will skip tests but still pass. Add tests as you develop features.

## Verification

After setup, verify everything works:

1. **Test Branch Protection**:
   ```bash
   git checkout -b test/branch-protection
   echo "test" > test.txt
   git add test.txt
   git commit -m "test: verify branch protection"
   git push origin test/branch-protection
   ```

2. **Create a test PR** - it should:
   - Show the PR template
   - Require status checks to pass
   - Block merging until approved

3. **Check Labels** - create a test issue and verify labels are available

## Maintenance

- Review and update labels as your project evolves
- Adjust branch protection rules based on team size and workflow
- Monitor CI workflow performance and adjust as needed
